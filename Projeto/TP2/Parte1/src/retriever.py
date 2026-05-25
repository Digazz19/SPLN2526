"""
retriever.py
------------
Módulo de retrieval com três modos:
  - lexical  : TF-IDF + cosine similarity (sklearn)
  - semantic : SBERT + cosine similarity  (sentence-transformers)
  - hybrid   : combinação linear dos dois scores

Opcionalmente, aplica cross-encoder re-ranking após o retrieval inicial.

Uso rápido:
    from retriever import Retriever
    r = Retriever("corpus.json")
    results = r.search("Who scored the winning goal in the 2022 World Cup?", mode="hybrid", top_k=5)
    for doc, score in results:
        print(f"[{score:.3f}] {doc['title']}")

Requisitos:
    pip install scikit-learn sentence-transformers torch numpy
"""

import json
import pickle
import os
import numpy as np
from typing import Literal

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
SBERT_MODEL         = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CACHE_TFIDF         = "cache_tfidf.pkl"
CACHE_SBERT         = "cache_sbert.pt"
RERANK_CANDIDATES   = 20    # candidatos passados ao cross-encoder
RERANK_CONTEXT_LEN  = 512   # chars do texto usados no re-ranking


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class Retriever:
    """
    Pipeline de retrieval léxico + semântico + re-ranking sobre o corpus.

    Parameters
    ----------
    corpus_path : str   — caminho para corpus.json
    sbert_model : str   — modelo SBERT a usar
    use_cache   : bool  — guarda/lê índices em disco
    """

    def __init__(
        self,
        corpus_path: str = "corpus.json",
        sbert_model: str = SBERT_MODEL,
        use_cache: bool = True,
    ):
        print("🔄 A carregar corpus...")
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

        self.texts     = [doc["text"]  for doc in self.corpus]
        self.titles    = [doc["title"] for doc in self.corpus]
        self.use_cache = use_cache
        self.cross_encoder = None   # carregado on-demand

        self._build_tfidf_index()
        self._build_sbert_index(sbert_model)
        print(" Retriever pronto.\n")

    # ── TF-IDF index ──────────────────────────────────────────────────────────

    def _build_tfidf_index(self) -> None:
        if self.use_cache and os.path.exists(CACHE_TFIDF):
            print("  [TF-IDF] A carregar cache...")
            with open(CACHE_TFIDF, "rb") as f:
                data = pickle.load(f)
            self.tfidf_vectorizer = data["vectorizer"]
            self.tfidf_matrix     = data["matrix"]
        else:
            print("  [TF-IDF] A construir índice...")
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_df=0.85,
                min_df=2,
                sublinear_tf=True,
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.texts)
            if self.use_cache:
                with open(CACHE_TFIDF, "wb") as f:
                    pickle.dump(
                        {"vectorizer": self.tfidf_vectorizer, "matrix": self.tfidf_matrix},
                        f,
                    )
        print(f"  [TF-IDF] {self.tfidf_matrix.shape[0]} docs × {self.tfidf_matrix.shape[1]} termos")

    # ── SBERT index ───────────────────────────────────────────────────────────

    def _build_sbert_index(self, model_name: str) -> None:
        if self.use_cache and os.path.exists(CACHE_SBERT):
            print("  [SBERT]  A carregar cache...")
            self.sbert_embeddings = torch.load(CACHE_SBERT)
        else:
            print(f"  [SBERT]  A carregar modelo '{model_name}'...")
            self.sbert_model = SentenceTransformer(model_name)
            print("  [SBERT]  A codificar corpus...")
            self.sbert_embeddings = self.sbert_model.encode(
                self.texts,
                convert_to_tensor=True,
                show_progress_bar=True,
                batch_size=32,
            )
            if self.use_cache:
                torch.save(self.sbert_embeddings, CACHE_SBERT)

        if not hasattr(self, "sbert_model"):
            self.sbert_model = SentenceTransformer(model_name)

        print(f"  [SBERT]  {self.sbert_embeddings.shape[0]} embeddings × dim {self.sbert_embeddings.shape[1]}")

    # ── Cross-Encoder re-ranking ───────────────────────────────────────────────

    def _load_cross_encoder(self) -> None:
        """Carrega o cross-encoder na primeira vez que é necessário."""
        if self.cross_encoder is None:
            print(f"  [CrossEncoder] A carregar '{CROSS_ENCODER_MODEL}'...")
            self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
            print("  [CrossEncoder] Pronto.")

    def _rerank(self, query: str, candidates: list, top_k: int) -> list:
        """
        Re-ranking com cross-encoder.

        Ao contrário do bi-encoder (SBERT) que codifica query e documento
        separadamente, o cross-encoder processa o par (query, documento)
        em conjunto, permitindo interação direta entre todos os tokens.
        Muito mais preciso, mas não escalável para corpora grandes.

        Parameters
        ----------
        query      : str   — pergunta do utilizador
        candidates : list  — lista de (doc, score) do retriever inicial
        top_k      : int   — quantos documentos retornar após re-ranking
        """
        self._load_cross_encoder()

        # Prepara pares (query, contexto truncado)
        pairs = [
            (query, doc["text"][:RERANK_CONTEXT_LEN])
            for doc, _ in candidates
        ]

        # Cross-encoder avalia cada par em conjunto
        cross_scores = self.cross_encoder.predict(pairs)

        # Re-ordena pelos scores do cross-encoder
        reranked = sorted(
            zip([doc for doc, _ in candidates], cross_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(doc, float(score)) for doc, score in reranked[:top_k]]

    # ── Métodos de pesquisa ───────────────────────────────────────────────────

    def _search_tfidf(self, query: str) -> np.ndarray:
        q_vec  = self.tfidf_vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        return scores

    def _search_sbert(self, query: str) -> np.ndarray:
        q_emb  = self.sbert_model.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(q_emb, self.sbert_embeddings).cpu().numpy().flatten()
        return scores

    def search(
        self,
        query: str,
        mode: Literal["lexical", "semantic", "hybrid"] = "hybrid",
        top_k: int = 5,
        alpha: float = 0.5,
        use_reranker: bool = False,
    ) -> list:
        """
        Pesquisa os documentos mais relevantes para a query.

        Parameters
        ----------
        query        : str   — pergunta ou texto de pesquisa
        mode         : str   — "lexical" | "semantic" | "hybrid"
        top_k        : int   — número de documentos a retornar
        alpha        : float — peso do TF-IDF no hybrid (0=só SBERT, 1=só TF-IDF)
        use_reranker : bool  — aplica cross-encoder re-ranking após retrieval
        """
        if mode == "lexical":
            scores = self._search_tfidf(query)

        elif mode == "semantic":
            scores = self._search_sbert(query)

        elif mode == "hybrid":
            tfidf_scores = _min_max_norm(self._search_tfidf(query))
            sbert_scores = _min_max_norm(self._search_sbert(query))
            scores = alpha * tfidf_scores + (1 - alpha) * sbert_scores

        else:
            raise ValueError(f"mode deve ser 'lexical', 'semantic' ou 'hybrid'. Recebido: '{mode}'")

        # Com re-ranking, recupera mais candidatos para depois filtrar
        n_candidates = RERANK_CANDIDATES if use_reranker else top_k
        top_indices  = np.argsort(scores)[::-1][:n_candidates]
        candidates   = [(self.corpus[i], float(scores[i])) for i in top_indices]

        if use_reranker:
            candidates = self._rerank(query, candidates, top_k)

        return candidates


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def _min_max_norm(arr: np.ndarray) -> np.ndarray:
    min_v, max_v = arr.min(), arr.max()
    if max_v - min_v == 0:
        return np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)


def print_results(results: list, query: str, mode: str) -> None:
    print(f"\n🔍 Query   : \"{query}\"")
    print(f"   Modo    : {mode}")
    print(f"   Top {len(results)} documentos:")
    print("   " + "─" * 60)
    for i, (doc, score) in enumerate(results, 1):
        snippet = doc["text"][:150].replace("\n", " ")
        print(f"   {i}. [{score:.3f}] {doc['title']} ({doc['category']})")
        print(f"         {snippet}...")
    print()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    retriever = Retriever("corpus.json")

    queries = [
        ("Who is Cristiano Ronaldo?",               "semantic"),
        ("Champions League winners history",         "lexical"),
        ("Which stadium has the largest capacity?",  "hybrid"),
        ("Messi World Cup Argentina 2022",           "hybrid"),
        ("Penalty kick rules offside",               "lexical"),
    ]

    print("=== SEM RE-RANKING ===")
    for query, mode in queries:
        results = retriever.search(query, mode=mode, top_k=3)
        print_results(results, query, mode)

    print("\n=== COM CROSS-ENCODER RE-RANKING ===")
    for query, mode in queries:
        results = retriever.search(query, mode=mode, top_k=3, use_reranker=True)
        print_results(results, query, f"{mode} + rerank")