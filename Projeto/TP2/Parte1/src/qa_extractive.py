"""
qa_extractive.py
----------------
Módulo de QA extrativo baseado em BERT fine-tuned no SQuAD v1.1.

Duas funcionalidades:
  1. train()     — faz fine-tune de bert-base-uncased no SQuAD v1.1 e guarda o modelo
  2. predict()   — dado um contexto (documento) e uma pergunta, extrai a resposta

Como o BERT tem limite de 512 tokens, documentos longos são partidos em
chunks com overlap antes da inferência (sliding window).

Uso típico:
    # 1. Treinar uma vez (≈1h com RTX 3060, dataset completo)
    python qa_extractive.py --train

    # 2. Testar inferência
    python qa_extractive.py --predict

Requisitos:
    pip install transformers datasets accelerate evaluate torch
"""

import argparse
import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
BASE_MODEL   = "bert-base-uncased"
MODEL_OUTPUT = "models/bert-squad"      # onde o modelo fine-tuned é guardado
MAX_LENGTH   = 384                      # tokens máximos por chunk
DOC_STRIDE   = 128                      # overlap entre chunks (sliding window)

# ---------------------------------------------------------------------------
# 1. PRÉ-PROCESSAMENTO DO SQuAD
# ---------------------------------------------------------------------------

def preprocess_training_examples(examples, tokenizer):
    """
    Tokeniza os exemplos do SQuAD e calcula as posições start/end
    da resposta em termos de tokens (não de caracteres).
    Implementa sliding window para contextos longos.
    """
    questions = [q.strip() for q in examples["question"]]

    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping   = inputs.pop("offset_mapping")
    sample_map       = inputs.pop("overflow_to_sample_mapping")
    answers          = examples["answers"]
    start_positions  = []
    end_positions    = []

    for i, offset in enumerate(offset_mapping):
        sample_idx    = sample_map[i]
        answer        = answers[sample_idx]
        sequence_ids  = inputs.sequence_ids(i)

        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        start_char = answer["answer_start"][0]
        end_char   = start_char + len(answer["text"][0])

        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"]   = end_positions
    return inputs


# ---------------------------------------------------------------------------
# 2. TREINO
# ---------------------------------------------------------------------------

def train():
    print("A carregar SQuAD v1.1...")
    raw_datasets = load_dataset("squad")

    print(f"   Train: {len(raw_datasets['train'])} exemplos")
    print(f"   Val  : {len(raw_datasets['validation'])} exemplos")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForQuestionAnswering.from_pretrained(BASE_MODEL)

    print("A pré-processar dataset...")
    train_dataset = raw_datasets["train"].map(
        lambda ex: preprocess_training_examples(ex, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    print("A iniciar fine-tuning...")
    args = TrainingArguments(
        output_dir=MODEL_OUTPUT,
        eval_strategy="no",         # sem avaliação durante treino
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,         # reduz para 1 se quiseres treino mais rápido (~30min)
        weight_decay=0.01,
        fp16=True,                  # FP16 para RTX 3060
        load_best_model_at_end=False,
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DefaultDataCollator(),
    )

    trainer.train()

    print(f"Modelo guardado em '{MODEL_OUTPUT}'")
    tokenizer.save_pretrained(MODEL_OUTPUT)
    model.save_pretrained(MODEL_OUTPUT)


# ---------------------------------------------------------------------------
# 3. INFERÊNCIA (com sliding window para documentos longos)
# ---------------------------------------------------------------------------

class ExtractiveQA:
    """
    Carrega o modelo fine-tuned e responde a perguntas sobre um contexto.
    Suporta documentos longos via sliding window (chunking).
    """

    def __init__(self, model_path: str = MODEL_OUTPUT):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modelo não encontrado em '{model_path}'.\n"
                f"Corre primeiro: python qa_extractive.py --train"
            )
        print(f"A carregar modelo de '{model_path}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model     = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"   Dispositivo: {self.device}")

    def predict(self, question: str, context: str, top_k: int = 1) -> list:
        """
        Responde a uma pergunta dado um contexto (documento).
        Usa sliding window se o contexto for maior que MAX_LENGTH tokens.

        Returns
        -------
        Lista de dicts com 'answer', 'score', 'start', 'end'
        ordenada por score decrescente.
        """
        inputs = self.tokenizer(
            question,
            context,
            max_length=MAX_LENGTH,
            truncation="only_second",
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )

        offset_mapping = inputs.pop("offset_mapping")
        inputs.pop("overflow_to_sample_mapping")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        start_logits = outputs.start_logits.cpu()
        end_logits   = outputs.end_logits.cpu()

        candidates = []

        for i in range(len(start_logits)):
            offsets = offset_mapping[i]

            # Localiza início e fim do contexto (tokens com offset não-None e não-(0,0))
            ctx_start, ctx_end = None, None
            for j, off in enumerate(offsets):
                if off is not None and list(off) != [0, 0]:
                    if ctx_start is None:
                        ctx_start = j
                    ctx_end = j

            if ctx_start is None:
                continue

            start_log = start_logits[i][ctx_start:ctx_end + 1]
            end_log   = end_logits[i][ctx_start:ctx_end + 1]

            scores = start_log.unsqueeze(1) + end_log.unsqueeze(0)
            mask   = torch.triu(torch.ones_like(scores), diagonal=0)
            scores = scores * mask + (1 - mask) * (-1e9)

            best_idx   = scores.argmax()
            best_start = best_idx // scores.shape[1]
            best_end   = best_idx  % scores.shape[1]
            score      = scores[best_start, best_end].item()

            abs_start = best_start.item() + ctx_start
            abs_end   = best_end.item()   + ctx_start

            if offsets[abs_start] is None or offsets[abs_end] is None:
                continue

            char_start = offsets[abs_start][0]
            char_end   = offsets[abs_end][1]
            answer     = context[char_start:char_end]

            if answer.strip():
                candidates.append({
                    "answer": answer.strip(),
                    "score":  score,
                    "start":  char_start,
                    "end":    char_end,
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]


# ---------------------------------------------------------------------------
# 4. DEMO
# ---------------------------------------------------------------------------

def demo_predict():
    qa = ExtractiveQA()

    examples = [
        {
            "question": "Where was Cristiano Ronaldo born?",
            "context": (
                "Cristiano Ronaldo dos Santos Aveiro was born on 5 February 1985 "
                "in Funchal, Madeira, Portugal. He is a Portuguese professional "
                "footballer who plays as a forward."
            ),
        },
        {
            "question": "How many Ballon d'Or awards has Messi won?",
            "context": (
                "Lionel Messi has won the Ballon d'Or a record eight times, in "
                "2009, 2010, 2011, 2012, 2019, 2021, 2023, and 2023. He is widely "
                "regarded as one of the greatest footballers of all time."
            ),
        },
        {
            "question": "When was the UEFA Champions League founded?",
            "context": (
                "The UEFA Champions League is an annual club football competition "
                "organised by UEFA. It was founded in 1955 as the European Cup and "
                "rebranded as the UEFA Champions League in 1992."
            ),
        },
    ]

    print("\n" + "=" * 60)
    print("  DEMO — QA Extrativo (BERT fine-tuned no SQuAD v1.1)")
    print("=" * 60)

    for ex in examples:
        results = qa.predict(ex["question"], ex["context"])
        print(f"\n {ex['question']}")
        if results:
            print(f"Resposta : {results[0]['answer']}")
            print(f"   Score   : {results[0]['score']:.2f}")
        else:
            print("Sem resposta encontrada.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QA Extrativo com BERT + SQuAD")
    parser.add_argument("--train",   action="store_true", help="Fine-tune BERT no SQuAD v1.1")
    parser.add_argument("--predict", action="store_true", help="Corre demo de inferência")
    args = parser.parse_args()

    if args.train:
        train()
    if args.predict:
        demo_predict()
    if not args.train and not args.predict:
        print("Usa --train para treinar ou --predict para inferência.")
        print("Exemplo: python qa_extractive.py --train")