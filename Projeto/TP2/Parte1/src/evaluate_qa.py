"""
evaluate_qa.py
--------------
Avaliação quantitativa dos módulos de QA.

Métricas implementadas:
  - Extrativo : Exact Match (EM) e F1 token-level (standard SQuAD)
  - Abstractivo: ROUGE-1, ROUGE-2, ROUGE-L

O script cria um conjunto de pares (pergunta, contexto, resposta_esperada)
a partir do corpus de futebol e avalia ambos os modelos.

Uso:
    python evaluate_qa.py

Requisitos:
    pip install rouge-score
"""

import json
import string
import re
from collections import Counter
from rouge_score import rouge_scorer

from qa_extractive  import ExtractiveQA
from qa_abstractive import AbstractiveQA

# ---------------------------------------------------------------------------
# Dataset de avaliação — pares (pergunta, título_doc, resposta_esperada)
# Construído manualmente a partir do corpus de futebol
# ---------------------------------------------------------------------------

EVAL_SET = [
    # Jogadores
    {
        "question": "Where was Cristiano Ronaldo born?",
        "doc_title": "Cristiano Ronaldo",
        "answer":    "Funchal, Madeira",
    },
    {
        "question": "What nationality is Lionel Messi?",
        "doc_title": "Lionel Messi",
        "answer":    "Argentine",
    },
    {
        "question": "When was Pelé born?",
        "doc_title": "Pelé",
        "answer":    "23 October 1940",
    },
    {
        "question": "What position does Mohamed Salah play?",
        "doc_title": "Mohamed Salah",
        "answer":    "forward",
    },
    {
        "question": "What year was Ronaldinho born?",
        "doc_title": "Ronaldinho",
        "answer":    "1980",
    },
    {
        "question": "What club did Zinedine Zidane play for in France?",
        "doc_title": "Zinedine Zidane",
        "answer":    "Girondins de Bordeaux",
    },
    {
        "question": "How tall is Erling Haaland?",
        "doc_title": "Erling Haaland",
        "answer":    "194 cm",
    },
    {
        "question": "What is David Beckham's nationality?",
        "doc_title": "David Beckham",
        "answer":    "English",
    },
    # Clubes
    {
        "question": "When was FC Barcelona founded?",
        "doc_title": "FC Barcelona",
        "answer":    "1899",
    },
    {
        "question": "What city is Bayern Munich from?",
        "doc_title": "Bayern Munich",
        "answer":    "Munich",
    },
    {
        "question": "What league does Sporting CP play in?",
        "doc_title": "Sporting CP",
        "answer":    "Primeira Liga",
    },
    {
        "question": "What is the nickname of Borussia Dortmund?",
        "doc_title": "Borussia Dortmund",
        "answer":    "BVB",
    },
    # Competições
    {
        "question": "How often is the FIFA World Cup held?",
        "doc_title": "FIFA World Cup",
        "answer":    "four years",
    },
    {
        "question": "When was the UEFA Champions League rebranded from the European Cup?",
        "doc_title": "UEFA Champions League",
        "answer":    "1992",
    },
    {
        "question": "What country hosts the Copa América?",
        "doc_title": "Copa América",
        "answer":    "South America",
    },
    # Estádios
    {
        "question": "What club plays at Camp Nou?",
        "doc_title": "Camp Nou",
        "answer":    "FC Barcelona",
    },
    {
        "question": "In what city is Anfield located?",
        "doc_title": "Anfield",
        "answer":    "Liverpool",
    },
    {
        "question": "When was Wembley Stadium reopened after reconstruction?",
        "doc_title": "Wembley Stadium",
        "answer":    "2007",
    },
    # Conceitos
    {
        "question": "What law governs offside in football?",
        "doc_title": "Offside (association football)",
        "answer":    "Law 11",
    },
    {
        "question": "What does VAR stand for?",
        "doc_title": "VAR (association football)",
        "answer":    "Video Assistant Referee",
    },
]


# ---------------------------------------------------------------------------
# Métricas SQuAD — Exact Match e F1
# (implementação oficial do SQuAD evaluation script)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lowercase, remove pontuação, artigos e espaços extra."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens  = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall    = num_same / len(truth_tokens)
    f1        = (2 * precision * recall) / (precision + recall)
    return f1


# ---------------------------------------------------------------------------
# Métricas ROUGE para QA abstractivo
# ---------------------------------------------------------------------------

def rouge_scores(prediction: str, ground_truth: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ground_truth, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


# ---------------------------------------------------------------------------
# Carrega o contexto do documento a partir do corpus
# ---------------------------------------------------------------------------

def load_corpus(path: str = "corpus.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    return {doc["title"]: doc["text"] for doc in corpus}


# ---------------------------------------------------------------------------
# Avaliação principal
# ---------------------------------------------------------------------------

def evaluate():
    print("A carregar corpus e modelos...")
    corpus      = load_corpus()
    extractive  = ExtractiveQA()
    abstractive = AbstractiveQA()

    ext_em_scores  = []
    ext_f1_scores  = []
    abs_rouge1     = []
    abs_rouge2     = []
    abs_rougeL     = []

    results = []

    print(f"\nA avaliar {len(EVAL_SET)} exemplos...\n")
    print("─" * 80)

    for i, example in enumerate(EVAL_SET, 1):
        question   = example["question"]
        doc_title  = example["doc_title"]
        expected   = example["answer"]

        # Obtém o contexto do corpus
        if doc_title not in corpus:
            print(f"  [{i:02d}] Documento '{doc_title}' não encontrado no corpus.")
            continue

        context = corpus[doc_title]

        # ── QA Extrativo ──
        ext_result = extractive.predict(question, context, top_k=1)
        ext_answer = ext_result[0]["answer"] if ext_result else ""

        em = exact_match(ext_answer, expected)
        f1 = f1_score(ext_answer, expected)
        ext_em_scores.append(em)
        ext_f1_scores.append(f1)

        # ── QA Abstractivo ──
        abs_answer = abstractive.predict(question, context)
        rouge = rouge_scores(abs_answer, expected)
        abs_rouge1.append(rouge["rouge1"])
        abs_rouge2.append(rouge["rouge2"])
        abs_rougeL.append(rouge["rougeL"])

        results.append({
            "question":    question,
            "expected":    expected,
            "extractive":  ext_answer,
            "abstractive": abs_answer,
            "em":          em,
            "f1":          round(f1, 3),
            "rouge1":      round(rouge["rouge1"], 3),
            "rougeL":      round(rouge["rougeL"], 3),
        })

        # Print por linha
        em_icon = "✅" if em else "❌"
        print(f"  [{i:02d}] {question}")
        print(f"        Expected   : {expected}")
        print(f"        Extrativo  : {ext_answer}  {em_icon}  EM={em}  F1={f1:.2f}")
        print(f"        Abstractivo: {abs_answer}  ROUGE-1={rouge['rouge1']:.2f}  ROUGE-L={rouge['rougeL']:.2f}")
        print()

    # ── Resultados globais ──
    n = len(results)
    avg_em    = sum(ext_em_scores)  / n * 100
    avg_f1    = sum(ext_f1_scores)  / n * 100
    avg_r1    = sum(abs_rouge1)     / n * 100
    avg_r2    = sum(abs_rouge2)     / n * 100
    avg_rL    = sum(abs_rougeL)     / n * 100

    print("=" * 80)
    print(f"  RESULTADOS GLOBAIS ({n} exemplos)")
    print("=" * 80)
    print(f"\n  QA Extrativo (BERT fine-tuned no SQuAD v1.1):")
    print(f"    Exact Match : {avg_em:.1f}%")
    print(f"    F1 Score    : {avg_f1:.1f}%")
    print(f"\n  QA Abstractivo (Flan-T5-large):")
    print(f"    ROUGE-1     : {avg_r1:.1f}%")
    print(f"    ROUGE-2     : {avg_r2:.1f}%")
    print(f"    ROUGE-L     : {avg_rL:.1f}%")
    print()

    # Guarda resultados em JSON para o relatório
    output = {
        "n_examples": n,
        "extractive": {
            "exact_match": round(avg_em, 1),
            "f1":          round(avg_f1, 1),
        },
        "abstractive": {
            "rouge1": round(avg_r1, 1),
            "rouge2": round(avg_r2, 1),
            "rougeL": round(avg_rL, 1),
        },
        "examples": results,
    }

    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("💾 Resultados guardados em 'evaluation_results.json'")
    return output


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    evaluate()