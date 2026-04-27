"""
tfidf_ir.py
===========
Sistema de Recuperação de Informação (IR) baseado em TF-IDF aplicado a um
corpus de documentos em língua inglesa.

Implementação manual (sem frameworks de IR externas) usando apenas:
  - nltk          (tokenização + stopwords)
  - collections   (Counter)
  - math          (log, sqrt)

Pipeline:
  1. Tokenização do corpus com remoção de stopwords
  2. Cálculo do TF por documento (normalizado pelo número de tokens)
  3. Cálculo do IDF com base logarítmica (log10(N/df))
  4. Construção da matriz TF-IDF e vetorização
  5. Recuperação por similaridade cosseno entre query e documentos
"""

import math
from collections import Counter

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


CORPUS = [
    "the sky is blue",
    "the sun is bright",
    "the sun in the sky",
]

QUERY = "The bright sun"


def tokenizer(doc: str) -> list[str]:
    """Tokeniza um documento e remove stopwords em inglês."""
    return [w for w in word_tokenize(doc) if w not in stopwords.words("english")]


def tf(t: str, d: list[str]) -> float:
    """TF de um termo num documento (frequência relativa)."""
    N = len(d)
    count = sum(1 for word in d if word == t)
    return count / N


def doc_tf(doc: list[str]) -> Counter:
    """Devolve o TF de todos os termos de um documento, num Counter."""
    N = len(doc)
    counter = Counter(doc)
    for c in counter:
        counter[c] = counter[c] / N
    return counter


def idf(corpus_tokens: list[list[str]]) -> dict[str, float]:
    """IDF de cada termo do corpus, com log de base 10."""
    N = len(corpus_tokens)
    res = {}
    for d in corpus_tokens:
        for t in set(d):
            if t not in res:
                df = sum(1 for doc in corpus_tokens if t in doc)
                res[t] = math.log(N / df, 10)
    return res


def tf_idf(corpus_tokens: list[list[str]]) -> list[dict[str, float]]:
    """Matriz TF-IDF como lista de dicionários (um por documento)."""
    idf_values = idf(corpus_tokens)
    matrix = []
    for doc in corpus_tokens:
        d = {}
        tf_values = doc_tf(doc)
        for t in tf_values:
            d[t] = tf_values[t] * idf_values[t]
        matrix.append(d)
    return matrix


def vectorize(tf_idf_dict: list[dict[str, float]],
              corpus_tokens: list[list[str]]) -> list[list[float]]:
    """Converte a matriz TF-IDF (dicts) em vetores densos sobre o vocabulário."""
    vocab = set(token for d in corpus_tokens for token in d)
    res = []
    for doc in tf_idf_dict:
        res.append([doc.get(token, 0) for token in vocab])
    return res


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Similaridade cosseno entre dois vetores."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a ** 2 for a in vec_a))
    norm_b = math.sqrt(sum(b ** 2 for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def vectorize_query(query_tokens: list[str],
                    corpus_tokens: list[list[str]],
                    idf_values: dict[str, float]) -> tuple[list[float], list[str]]:
    """Vetoriza a query no mesmo espaço do vocabulário do corpus.
    Palavras fora do vocabulário recebem IDF = 0."""
    vocab = list(set(token for d in corpus_tokens for token in d))
    tf_values = doc_tf(query_tokens)
    vec = []
    for token in vocab:
        tf_val = tf_values.get(token, 0)
        idf_val = idf_values.get(token, 0)
        vec.append(tf_val * idf_val)
    return vec, vocab


def main():
    # 1. Tokenização
    corpus_tokens = [tokenizer(doc) for doc in CORPUS]
    print("Corpus tokenizado:")
    print(corpus_tokens)
    print()

    # 2. TF de exemplo (primeiro documento)
    print(f"TF do primeiro documento: {doc_tf(corpus_tokens[0])}")
    print()

    # 3. IDF do corpus
    idf_values = idf(corpus_tokens)
    print(f"IDF do corpus: {idf_values}")
    print()

    # 4. Matriz TF-IDF
    tfidf_matrix = tf_idf(corpus_tokens)
    print("Matriz TF-IDF (por documento):")
    for i, d in enumerate(tfidf_matrix):
        print(f"  Doc {i}: {d}")
    print()

    # 5. Vetorização
    doc_vecs = vectorize(tfidf_matrix, corpus_tokens)
    print("Vetores densos:")
    for i, v in enumerate(doc_vecs):
        print(f"  Doc {i}: {v}")
    print()

    # 6. Information Retrieval da query
    print(f"Query: {QUERY!r}")
    query_tokens = tokenizer(QUERY)
    print(f"Query tokenizada: {query_tokens}")
    print()

    query_vec, _vocab = vectorize_query(query_tokens, corpus_tokens, idf_values)

    scores = []
    for i, doc_vec in enumerate(doc_vecs):
        score = cosine_similarity(query_vec, doc_vec)
        scores.append((i, score, CORPUS[i]))

    # Ordenar por relevância (decrescente)
    scores.sort(key=lambda x: x[1], reverse=True)

    print("Resultados (ordenados por relevância):")
    for rank, (doc_idx, score, doc_text) in enumerate(scores, 1):
        print(f'  {rank}. [score={score:.4f}] Doc {doc_idx}: "{doc_text}"')


if __name__ == "__main__":
    main()