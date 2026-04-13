import re
import os
import logging
import warnings
import spacy
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec, Phrases
from gensim.models.phrases import Phraser
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

BOOKS_DIR  = "livros"               # Pasta com todos os .txt dos livros
MODEL_DIR  = "models"
PLOTS_DIR  = "plots"
MODEL_PATH = os.path.join(MODEL_DIR, "hp_word2vec.model")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. PRÉ-PROCESSAMENTO COM spaCy
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(text: str, nlp) -> list[list[str]]:
    """
    Tokeniza e lematiza o texto com spaCy.
    Remove stopwords, pontuação e números.
    Devolve lista de frases, cada frase = lista de lemas.
    """
    sentences = []
    # Processar em batches para eficiência
    for doc in nlp.pipe(text.split("\n"), batch_size=64, disable=["ner", "parser"]):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.like_num
            and token.is_alpha
            and len(token.text) > 1
        ]
        if tokens:
            sentences.append(tokens)
    return sentences


def detect_ngrams(sentences: list[list[str]]) -> list[list[str]]:
    """
    Deteta bigramas e trigramas automaticamente.
    Ex: ['harry', 'potter'] → ['harry_potter']
    """
    bigram_model  = Phrases(sentences, min_count=5, threshold=5)
    trigram_model = Phrases(bigram_model[sentences], min_count=3, threshold=5)

    bigram  = Phraser(bigram_model)
    trigram = Phraser(trigram_model)

    sentences_bi  = [bigram[s]  for s in sentences]
    sentences_tri = [trigram[s] for s in sentences_bi]

    # Mostrar exemplos de n-grams detetados
    ngrams_found = set()
    for s in sentences_tri:
        for token in s:
            if "_" in token:
                ngrams_found.add(token)
    print(f"\n[N-Grams] {len(ngrams_found)} n-grams detetados. Exemplos:")
    for ng in sorted(ngrams_found)[:15]:
        print(f"  • {ng}")

    return sentences_tri


# ─────────────────────────────────────────────────────────────────────────────
# 2. TREINO DO MODELO WORD2VEC
# ─────────────────────────────────────────────────────────────────────────────

def train_model(sentences: list[list[str]]) -> Word2Vec:
    """
    Treina o modelo Word2Vec com Skip-Gram.
    Hiperparâmetros escolhidos para um corpus pequeno (1 livro).
    """
    model = Word2Vec(
        sentences,
        vector_size=150,  # mais dimensões — corpus maior suporta melhor
        window=5,         # contexto de 5 palavras
        min_count=5,      # com mais texto podemos ser mais exigentes
        sg=1,             # Skip-Gram (melhor para palavras raras/nomes próprios)
        workers=4,
        epochs=100,       # mais épocas → vetores mais estáveis
        seed=42
    )
    model.save(MODEL_PATH)
    print(f"\n[Modelo] Treinado com vocabulário de {len(model.wv)} palavras.")
    print(f"[Modelo] Guardado em '{MODEL_PATH}'")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3. ANÁLISES SEMÂNTICAS
# ─────────────────────────────────────────────────────────────────────────────

def run_most_similar(model: Word2Vec):
    """Most Similar — vizinhos mais próximos de termos chave."""
    print("\n" + "═"*60)
    print("  MOST SIMILAR")
    print("═"*60)

    queries = [
        "harry",
        "hermione",
        "ron",
        "dumbledore",
        "voldemort",
        "magic",
        "wand",
        "hogwarts",
        "quidditch",
        "potion",
    ]

    results = {}
    for word in queries:
        # Tentar também versão com n-gram
        candidates = [word, word + "_potter"] if word == "harry" else [word]
        for w in candidates:
            if w in model.wv:
                similar = model.wv.most_similar(w, topn=5)
                results[w] = similar
                print(f"\n  '{w}' → palavras mais próximas:")
                for term, score in similar:
                    print(f"      {term:<25} {score:.4f}")
                break
        else:
            print(f"\n  '{word}' — não encontrado no vocabulário")

    return results


def run_similarity(model: Word2Vec):
    """Similarity — similaridade direta entre pares de termos."""
    print("\n" + "═"*60)
    print("  SIMILARITY (cosseno)")
    print("═"*60)

    pairs = [
        ("harry",      "voldemort"),
        ("harry",      "hermione"),
        ("harry",      "ron"),
        ("hermione",   "ron"),
        ("dumbledore", "voldemort"),
        ("magic",      "wand"),
        ("hogwarts",   "school"),
        ("gryffindor", "slytherin"),
        ("potion",     "spell"),
        ("dragon",     "magic"),
    ]

    valid_pairs = []
    for w1, w2 in pairs:
        if w1 in model.wv and w2 in model.wv:
            score = model.wv.similarity(w1, w2)
            valid_pairs.append((w1, w2, score))
            bar = "█" * int(score * 20)
            print(f"  {w1:<15} ↔ {w2:<15}  {score:.4f}  {bar}")
        else:
            missing = [w for w in [w1, w2] if w not in model.wv]
            print(f"  {w1} ↔ {w2}  — ausente: {missing}")

    return valid_pairs


def run_doesnt_match(model: Word2Vec):
    """Doesn't Match — intruso num grupo de palavras."""
    print("\n" + "═"*60)
    print("  DOESN'T MATCH")
    print("═"*60)

    groups = [
        ["harry", "hermione", "ron", "dragon"],
        ["gryffindor", "slytherin", "hufflepuff", "london"],
        ["wand", "broomstick", "potion", "car"],
        ["dumbledore", "snape", "mcgonagall", "muggle"],
        ["magic", "spell", "enchantment", "stone"],
    ]

    for group in groups:
        # Filtrar palavras que existam no vocabulário
        valid = [w for w in group if w in model.wv]
        if len(valid) < 3:
            print(f"  {group}  → palavras insuficientes no vocabulário")
            continue
        odd = model.wv.doesnt_match(valid)
        print(f"  {valid}  →  intruso: '{odd}'")


def run_analogies(model: Word2Vec):
    """Analogias vetoriais: A está para B como C está para ?"""
    print("\n" + "═"*60)
    print("  ANALOGIAS  (A - B + C = ?)")
    print("═"*60)

    analogies = [
        # (positivo, positivo, negativo)  → esperado
        (["harry", "hermione"], ["ron"],  "trio de amigos"),
        (["dumbledore", "magic"], ["muggle"], "feiticeiro vs muggle"),
        (["wand", "spell"], ["potion"], "objeto mágico"),
    ]

    for pos, neg, label in analogies:
        all_words = pos + neg
        if all(w in model.wv for w in all_words):
            result = model.wv.most_similar(positive=pos, negative=neg, topn=3)
            print(f"\n  [{label}]  +{pos}  -{neg}")
            for term, score in result:
                print(f"      {term:<20} {score:.4f}")
        else:
            missing = [w for w in all_words if w not in model.wv]
            print(f"\n  [{label}] — ausente: {missing}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALIZAÇÕES
# ─────────────────────────────────────────────────────────────────────────────

CHARACTERS = [
    "harry", "hermione", "ron", "dumbledore", "voldemort",
    "snape", "hagrid", "neville", "draco", "quirrell"
]

CONCEPTS = [
    "magic", "wand", "spell", "potion", "broomstick",
    "hogwarts", "gryffindor", "slytherin", "quidditch", "dragon"
]


def get_vocab_words(model: Word2Vec, words: list[str]) -> list[str]:
    return [w for w in words if w in model.wv]


def plot_pca(model: Word2Vec):
    """Gráfico PCA 2D — personagens e conceitos mágicos."""
    chars    = get_vocab_words(model, CHARACTERS)
    concepts = get_vocab_words(model, CONCEPTS)
    words    = chars + concepts

    if len(words) < 4:
        print("[PCA] Palavras insuficientes no vocabulário para o gráfico.")
        return

    vectors = np.array([model.wv[w] for w in words])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor("#0a0a1a")
    fig.patch.set_facecolor("#0a0a1a")

    # Personagens a vermelho/dourado, conceitos a azul/verde
    for i, word in enumerate(words):
        color = "#FFD700" if word in chars else "#4FC3F7"
        marker = "" if word in chars else ""
        ax.scatter(coords[i, 0], coords[i, 1], color=color, s=80, zorder=3)
        ax.annotate(
            word, (coords[i, 0], coords[i, 1]),
            fontsize=9, color=color,
            textcoords="offset points", xytext=(6, 4),
            fontweight="bold"
        )

    var_explained = pca.explained_variance_ratio_
    ax.set_title(
        "Mapa Semântico PCA — Harry Potter e a Pedra Filosofal",
        color="white", fontsize=14, pad=15
    )
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)", color="gray")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)", color="gray")
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    # Legenda manual
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FFD700", label="Personagens"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4FC3F7", label="Conceitos Mágicos"),
    ]
    ax.legend(handles=legend, facecolor="#1a1a2e", labelcolor="white", framealpha=0.7)

    path = os.path.join(PLOTS_DIR, "pca_semantico.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[Gráfico] PCA guardado em '{path}'")


def plot_similarity_heatmap(model: Word2Vec, pairs_results: list):
    """Heatmap de similaridade entre os personagens principais."""
    chars = get_vocab_words(model, CHARACTERS)
    if len(chars) < 3:
        print("[Heatmap] Personagens insuficientes no vocabulário.")
        return

    n = len(chars)
    matrix = np.zeros((n, n))
    for i, w1 in enumerate(chars):
        for j, w2 in enumerate(chars):
            matrix[i][j] = model.wv.similarity(w1, w2)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        xticklabels=chars,
        yticklabels=chars,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        vmin=-1, vmax=1
    )
    ax.set_title(
        "Heatmap de Similaridade — Personagens Principais",
        fontsize=13, pad=12
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    path = os.path.join(PLOTS_DIR, "heatmap_personagens.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Gráfico] Heatmap guardado em '{path}'")


def plot_tsne(model: Word2Vec):
    """t-SNE com os top-50 termos mais frequentes."""
    words   = list(model.wv.index_to_key[:50])
    vectors = np.array([model.wv[w] for w in words])

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(15, len(words)-1))
    coords = tsne.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(coords[:, 0], coords[:, 1], s=40, color="#9C27B0", alpha=0.7)
    for i, word in enumerate(words):
        ax.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=7.5,
                    textcoords="offset points", xytext=(4, 2), color="#333")

    ax.set_title("t-SNE — Top 50 Palavras Mais Frequentes", fontsize=13)
    ax.set_xlabel("Dimensão 1")
    ax.set_ylabel("Dimensão 2")

    path = os.path.join(PLOTS_DIR, "tsne_top50.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Gráfico] t-SNE guardado em '{path}'")


def plot_most_similar_bar(model: Word2Vec):
    """Gráfico de barras com os top-8 similares de 'harry'."""
    target = "harry_potter" if "harry_potter" in model.wv else "harry"
    if target not in model.wv:
        print("[Gráfico] 'harry' não encontrado para o gráfico de barras.")
        return

    similar = model.wv.most_similar(target, topn=8)
    words_  = [w for w, _ in similar]
    scores  = [s for _, s in similar]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(words_[::-1], scores[::-1], color="#E65100", edgecolor="white", height=0.6)
    ax.set_xlabel("Similaridade (cosseno)")
    ax.set_title(f"Top 8 palavras mais similares a '{target}'", fontsize=12)
    ax.set_xlim(0, 1)
    for bar, score in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{score:.3f}", va="center", fontsize=9)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "most_similar_harry.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Gráfico] Barras most_similar guardado em '{path}'")

def export_tensorboard(model: Word2Vec, n_words: int = 500):
    """Exporta vetores e metadados para o TensorFlow Embedding Projector."""
    os.makedirs("Project_models", exist_ok=True)

    words = list(model.wv.index_to_key[:n_words])

    # Ficheiro de vetores (tensors.tsv)
    with open("Project_models/tensors.tsv", "w", encoding="utf-8") as f:
        for word in words:
            vector = model.wv[word]
            f.write("\t".join([str(x) for x in vector]) + "\n")

    # Ficheiro de metadados (metadata.tsv)
    with open("Project_models/metadata.tsv", "w", encoding="utf-8") as f:
        for word in words:
            f.write(word + "\n")

    print("[TensorBoard] Ficheiros exportados para 'Project_models/'")
    print(f"  tensors.tsv  — {n_words} vetores")
    print(f"  metadata.tsv — {n_words} palavras")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("  Harry Potter — Análise Semântica com Word2Vec + spaCy")
    print("=" * 60)

    # Carregar todos os livros da pasta
    if not os.path.isdir(BOOKS_DIR):
        raise FileNotFoundError(f"Pasta '{BOOKS_DIR}/' não encontrada.")

    book_files = sorted([
        f for f in os.listdir(BOOKS_DIR) if f.endswith(".txt")
    ])
    if not book_files:
        raise FileNotFoundError(f"Nenhum ficheiro .txt encontrado em '{BOOKS_DIR}/'.")

    print(f"[Livros] {len(book_files)} ficheiro(s) encontrado(s):")
    text = ""
    for fname in book_files:
        path = os.path.join(BOOKS_DIR, fname)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        text += content + "\n"
        print(f"  • {fname}  ({len(content):,} caracteres)")
    print(f"[Livros] Total: {len(text):,} caracteres combinados")

    # Carregar spaCy
    print("[spaCy] A carregar modelo pt_core_news_sm...")
    nlp = spacy.load("pt_core_news_sm")
    nlp.max_length = len(text) + 1000

    # Pré-processamento
    print("[spaCy] A tokenizar e lematizar...")
    sentences = preprocess(text, nlp)
    print(f"[spaCy] {len(sentences):,} frases processadas")

    # N-Grams
    sentences = detect_ngrams(sentences)

    # Treino
    if os.path.exists(MODEL_PATH):
        print(f"\n[Modelo] A carregar modelo existente de '{MODEL_PATH}'...")
        model = Word2Vec.load(MODEL_PATH)
    else:
        print("\n[Modelo] A treinar Word2Vec...")
        model = train_model(sentences)

    # ── Análises ──────────────────────────────────────────────────
    run_most_similar(model)
    pairs = run_similarity(model)
    run_doesnt_match(model)
    run_analogies(model)

    # ── Visualizações ─────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  A GERAR GRÁFICOS...")
    print("═"*60)
    plot_pca(model)
    plot_similarity_heatmap(model, pairs)
    plot_tsne(model)
    plot_most_similar_bar(model)
    export_tensorboard(model, n_words=500)

    print("\n  Análise concluída! Gráficos guardados em './plots/'")


if __name__ == "__main__":
    main()