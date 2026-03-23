import os
import re
import math
from collections import Counter

INPUT_DIR  = "textfiles_clean"
OUTPUT_DIR = "ngram_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N = 3 

SOURCES = [
    {
        "filename": os.path.join(INPUT_DIR, "historia_futebol_inglaterra_clean.txt"),
        "label":    "História do Futebol na Inglaterra",
    },
    {
        "filename": os.path.join(INPUT_DIR, "primeira_liga_clean.txt"),
        "label":    "Primeira Liga",
    },
    {
        "filename": os.path.join(INPUT_DIR, "revista_militar_futebol_clean.txt"),
        "label":    "Revista Militar – Geopolítica e Futebol",
    },
]

# Stopwords para português
STOP_PT = {
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com",
    "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como",
    "mas", "ao", "ele", "das", "à", "seu", "sua", "ou", "quando", "muito",
    "nos", "já", "eu", "também", "só", "pelo", "pela", "até", "isso",
    "ela", "entre", "depois", "sem", "mesmo", "aos", "seus", "quem",
    "nas", "me", "esse", "eles", "eram", "esta", "num", "nem", "suas",
    "meu", "minha", "numa", "foi", "ter", "há", "não", "está", "ser",
    "são", "tem", "este", "essa", "deste", "desta", "nesta", "neste",
    "nesse", "nessa", "aqui", "ali", "nós", "vós", "lhe", "lhes",
    "qual", "quais", "cujo", "cuja", "tudo", "todo", "toda", "todos",
    "todas", "outro", "outra", "outros", "outras", "tanto", "tanta",
}

# ── Tokenização ──────────────────────────────────────────────────────────────

def tokenize_sentences(text: str) -> list:
    """Divide o texto em frases usando pontuação final seguida de maiúscula."""
    frases = re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÀÂÊÔÃÕ])', text)
    return [f.strip() for f in frases if f.strip()]

def tokenize_words(sentence: str) -> list:
    """Extrai tokens alfabéticos e converte para minúsculas."""
    tokens = re.findall(r"[a-záéíóúàâêôãõüçA-ZÁÉÍÓÚÀÂÊÔÃÕÜÇ]+", sentence)
    return [t.lower() for t in tokens]

# ── Modelo de n-grams ────────────────────────────────────────────────────────

def build_ngram_model(sentences: list, n: int) -> dict:
    """
    Constrói um modelo de n-grams a partir de uma lista de frases.

    Para cada frase, os tokens são padded com marcadores <s> e </s>.
    Devolve:
      - ngram_counts:   Counter de n-grams (tuplo → contagem)
      - context_counts: Counter de (n-1)-grams (contexto → contagem)
      - vocab:          conjunto de todas as palavras vistas
      - n:              ordem do modelo
    """
    ngram_counts   = Counter()
    context_counts = Counter()
    vocab          = set()

    for sentence in sentences:
        tokens = tokenize_words(sentence)
        if not tokens:
            continue
        vocab.update(tokens)
        if len(tokens) < n:
            continue
        padded = ["<s>"] * (n - 1) + tokens + ["</s>"]
        for i in range(len(padded) - n + 1):
            ngram   = tuple(padded[i : i + n])
            context = tuple(padded[i : i + n - 1])
            ngram_counts[ngram]     += 1
            context_counts[context] += 1

    return {
        "ngram_counts":   ngram_counts,
        "context_counts": context_counts,
        "vocab":          vocab,
        "n":              n,
    }

def ngram_prob(model: dict, context: tuple, word: str) -> float:
    """
    Probabilidade condicional P(word | context) com suavização add-1 (Laplace).
    Garante que nunca há log(0).
    """
    vocab_size  = len(model["vocab"]) + 1  # +1 para </s>
    numerator   = model["ngram_counts"].get(context + (word,), 0) + 1
    denominator = model["context_counts"].get(context, 0) + vocab_size
    return numerator / denominator

# ── Scoring de frases ────────────────────────────────────────────────────────

def score_sentence(sentence: str, model: dict) -> float:
    """
    Pontua uma frase pela log-probabilidade média dos seus n-grams,
    calculada apenas sobre tokens que não são stopwords nem muito curtos.

    Frases com score mais alto contêm sequências de palavras mais frequentes
    no corpus, sendo portanto mais representativas do texto.
    """
    n      = model["n"]
    tokens = tokenize_words(sentence)
    tokens = [t for t in tokens if t not in STOP_PT and len(t) > 2]

    if len(tokens) < n:
        return float("-inf")

    padded   = ["<s>"] * (n - 1) + tokens + ["</s>"]
    log_prob = 0.0
    count    = 0

    for i in range(len(padded) - n + 1):
        context = tuple(padded[i : i + n - 1])
        word    = padded[i + n - 1]
        log_prob += math.log(ngram_prob(model, context, word))
        count    += 1

    return log_prob / count if count > 0 else float("-inf")

def select_top_sentences(sentences: list, model: dict, k: int = 3) -> list:
    """
    Seleciona as k frases com maior score, garantindo comprimento mínimo
    (evita frases demasiado curtas ou fragmentos).
    """
    MIN_WORDS = 8
    candidatas = [s for s in sentences if len(tokenize_words(s)) >= MIN_WORDS]
    scored = [(s, score_sentence(s, model)) for s in candidatas]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:k]]

# ── Pipeline principal ───────────────────────────────────────────────────────

def process_source(source: dict, n: int = N) -> dict:
    print(f"\n{'='*60}")
    print(f"Fonte: {source['label']}")
    print(f"{'='*60}")

    with open(source["filename"], "r", encoding="utf-8") as f:
        text = f.read()

    sentences = tokenize_sentences(text)
    print(f"  Frases encontradas : {len(sentences)}")

    model = build_ngram_model(sentences, n)
    print(f"  Vocabulário        : {len(model['vocab'])} palavras únicas")
    print(f"  N-grams únicos     : {len(model['ngram_counts'])} (n={n})")

    top = select_top_sentences(sentences, model, k=3)
    print(f"\n  ── 3 frases selecionadas ──")
    for i, s in enumerate(top, 1):
        print(f"\n  [{i}] {s}")

    return {
        "label":     source["label"],
        "sentences": sentences,
        "model":     model,
        "top":       top,
    }

def main():
    results = []
    for source in SOURCES:
        result = process_source(source, n=N)
        results.append(result)

    # Guardar frases selecionadas em ficheiro de texto
    out_path = os.path.join(OUTPUT_DIR, "frases_selecionadas.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"\n{'='*60}\n")
            f.write(f"{r['label']}\n")
            f.write(f"{'='*60}\n")
            for i, s in enumerate(r["top"], 1):
                f.write(f"\n[{i}] {s}\n")
    print(f"\n\nFrases guardadas em: {out_path}")

if __name__ == "__main__":
    main()