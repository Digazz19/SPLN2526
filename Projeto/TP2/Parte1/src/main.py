"""
main.py
-------
Pipeline completa de Information Retrieval + Question Answering.

Fluxo:
    query do utilizador
        → Retriever (TF-IDF / SBERT / hybrid)
            → top-K documentos relevantes
                → QA Extrativo  (BERT fine-tuned no SQuAD)
                → QA Abstractivo (Flan-T5-large via prompting)

Uso interativo:
    python main.py

Uso com argumentos:
    python main.py --mode hybrid --top_k 3 --query "Who won the 2022 World Cup?"

Requisitos:
    corpus.json         (gerado por corpus_builder.py)
    models/bert-squad/  (gerado por qa_extractive.py --train)
"""

import argparse
from retriever      import Retriever, print_results
from qa_extractive  import ExtractiveQA
from qa_abstractive import AbstractiveQA

# ---------------------------------------------------------------------------
# Configuração por defeito
# ---------------------------------------------------------------------------
DEFAULT_MODE   = "hybrid"
DEFAULT_TOP_K  = 3


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class FootballQAPipeline:
    """
    Pipeline completa: retrieval + QA extrativo + QA abstractivo.
    """

    def __init__(
        self,
        corpus_path: str = "corpus.json",
        retriever_mode: str = DEFAULT_MODE,
        top_k: int = DEFAULT_TOP_K,
    ):
        print("\n" + "=" * 65)
        print("Football QA Pipeline")
        print("=" * 65)

        self.retriever_mode = retriever_mode
        self.top_k          = top_k

        print("\n[1/3] A inicializar Retriever...")
        self.retriever = Retriever(corpus_path)

        print("\n[2/3] A inicializar QA Extrativo (BERT)...")
        self.extractive = ExtractiveQA()

        print("\n[3/3] A inicializar QA Abstractivo (Flan-T5)...")
        self.abstractive = AbstractiveQA()

        print("\nPipeline pronta!\n")

    def run(self, query: str) -> dict:
        """
        Executa a pipeline completa para uma query.

        Returns
        -------
        dict com:
            - query         : str
            - documents     : list[dict]  — documentos recuperados
            - extractive    : str         — resposta extrativa (melhor doc)
            - abstractive   : str         — resposta abstractiva (multi-doc)
        """
        print("\n" + "─" * 65)
        print(f"Query: {query}")
        print("─" * 65)

        # ── 1. Retrieval ──────────────────────────────────────────────
        print(f"\nRetrieval ({self.retriever_mode}, top-{self.top_k})...")
        results = self.retriever.search(
            query,
            mode=self.retriever_mode,
            top_k=self.top_k,
        )

        for i, (doc, score) in enumerate(results, 1):
            print(f"   {i}. [{score:.3f}] {doc['title']} ({doc['category']})")

        documents = [doc for doc, _ in results]

        # ── 2. QA Extrativo — usa o documento mais relevante ──────────
        print("\nQA Extrativo (BERT)...")
        best_doc     = documents[0]
        ext_results  = self.extractive.predict(query, best_doc["text"], top_k=1)

        if ext_results:
            ext_answer = ext_results[0]["answer"]
            ext_score  = ext_results[0]["score"]
            print(f"   Fonte   : {best_doc['title']}")
            print(f"   Resposta: {ext_answer}")
            print(f"   Score   : {ext_score:.2f}")
        else:
            ext_answer = "Não foi possível encontrar uma resposta."
            print(f"Sem resposta encontrada.")

        # ── 3. QA Abstractivo — usa o documento mais relevante ──────
        print("\nQA Abstractivo (Flan-T5)...")
        abs_answer = self.abstractive.predict(query, best_doc["text"])
        print(f"   Fonte   : {best_doc['title']}")
        print(f"   Resposta: {abs_answer}")

        # ── Resumo final ──────────────────────────────────────────────
        print("\n" + "┄" * 65)
        print(f"  EXTRATIVO  → {ext_answer}")
        print(f"  ABSTRACTIVO→ {abs_answer}")
        print("┄" * 65)

        return {
            "query":       query,
            "documents":   [{"title": d["title"], "category": d["category"]} for d in documents],
            "extractive":  ext_answer,
            "abstractive": abs_answer,
        }


# ---------------------------------------------------------------------------
# Modo interativo
# ---------------------------------------------------------------------------

def interactive_mode(pipeline: FootballQAPipeline) -> None:
    print("\nModo interativo — escreve 'sair' para terminar.")
    print("   Exemplos de queries:")
    print("   - Who won the 2022 FIFA World Cup?")
    print("   - What is the capacity of Camp Nou?")
    print("   - When was Cristiano Ronaldo born?")
    print("   - Which club has won the most Champions League titles?\n")

    while True:
        try:
            query = input("🔎 Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nA terminar.")
            break

        if not query:
            continue
        if query.lower() in ("sair", "exit", "quit"):
            print("A terminar.")
            break

        pipeline.run(query)


# ---------------------------------------------------------------------------
# Demo com queries predefinidas
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    "Who won the 2022 FIFA World Cup?",
    "What is the capacity of Camp Nou?",
    "When was Cristiano Ronaldo born?",
    "Which club has won the most Champions League titles?",
    "What are the offside rules in football?",
]


def demo_mode(pipeline: FootballQAPipeline) -> None:
    print("\nModo demo — a correr queries predefinidas...\n")
    for query in DEMO_QUERIES:
        pipeline.run(query)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Football QA Pipeline — IR + Extrativo + Abstractivo"
    )
    parser.add_argument(
        "--mode",
        choices=["lexical", "semantic", "hybrid"],
        default=DEFAULT_MODE,
        help="Modo de retrieval (default: hybrid)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Número de documentos a recuperar (default: 3)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query única (se não especificado, entra em modo interativo)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Corre queries de demonstração predefinidas",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="corpus.json",
        help="Caminho para o corpus (default: corpus.json)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pipeline = FootballQAPipeline(
        corpus_path=args.corpus,
        retriever_mode=args.mode,
        top_k=args.top_k,
    )

    if args.demo:
        demo_mode(pipeline)
    elif args.query:
        pipeline.run(args.query)
    else:
        interactive_mode(pipeline)