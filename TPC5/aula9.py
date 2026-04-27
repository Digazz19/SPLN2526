"""
aula9.py
============
Exercício da aula: fine-tuning de BERT
(`neuralmind/bert-base-portuguese-cased`) para NER em português, sobre o
dataset `lfcc/portuguese_ner` do HuggingFace.


Pipeline:
    1. Carregar dataset HuggingFace
    2. Tokenizar com BERT tokenizer (alinhando labels via word_ids)
    3. Fine-tune com Trainer (2 épocas, lr=2e-5, batch=16)
    4. Avaliação por época com seqeval (Precision / Recall / F1 / Accuracy)

Como correr:
    pip install -r requirementsAula.txt
    python aula9.py
"""

import json
from pathlib import Path

import evaluate
import numpy as np
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
DATASET_NAME = "lfcc/portuguese_ner"
CHECKPOINT = "neuralmind/bert-base-portuguese-cased"
OUTPUT_MODEL_NAME = "my_model"
NUM_EPOCHS = 2
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
WEIGHT_DECAY = 0.01
MAX_LENGTH = 512



def align_labels_with_tokens(labels: list[int], word_ids: list[int | None]) -> list[int]:
    """
    Alinha labels (uma por palavra) com sub-tokens do BERT.
    Sub-tokens não-iniciais e tokens especiais ([CLS], [SEP], padding) recebem
    -100 para serem ignorados pela cross-entropy loss.
    """
    new_labels = []
    previous_id = None
    for word_id in word_ids:
        if word_id is None:
            new_labels.append(-100)
        elif previous_id != word_id:
            # primeiro sub-token desta palavra -> recebe a label
            new_labels.append(labels[word_id])
        else:
            # sub-tokens seguintes da mesma palavra -> ignorados
            new_labels.append(-100)
        previous_id = word_id
    return new_labels


def tokenize_dataset(dataset, tokenizer):
    """Tokeniza um split do dataset HuggingFace e alinha as labels."""
    new_dataset = []
    for data in dataset:
        tokenized_inputs = tokenizer(
            data["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        new_labels = align_labels_with_tokens(
            data["ner_tags"], tokenized_inputs.word_ids()
        )
        tokenized_inputs["labels"] = new_labels
        new_dataset.append(tokenized_inputs)
    return new_dataset


def make_compute_metrics(label_list: list[str]):
    """Constrói a função compute_metrics com a lista de labels em closure."""
    
    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[pred] for (pred, lab) in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[lab] for (pred, lab) in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(
            predictions=true_predictions, references=true_labels
        )
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_metrics


def main():
    print(f"A carregar dataset '{DATASET_NAME}'...")
    raw_datasets = load_dataset(DATASET_NAME)
    print(raw_datasets)

    label_list = raw_datasets["train"].features["ner_tags"].feature.names
    print(f"  labels: {label_list}")

    print(f"\nA carregar tokenizer '{CHECKPOINT}'...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    print("\nA tokenizar splits train/test...")
    data_train = tokenize_dataset(raw_datasets["train"], tokenizer)
    data_test = tokenize_dataset(raw_datasets["test"], tokenizer)
    train_dataset = Dataset.from_list(data_train)
    test_dataset = Dataset.from_list(data_test)
    print(f"  treino: {train_dataset}")
    print(f"  teste:  {test_dataset}")

    print(f"\nA carregar modelo '{CHECKPOINT}'...")
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {v: k for k, v in label2id.items()}
    model = AutoModelForTokenClassification.from_pretrained(
        CHECKPOINT, id2label=id2label, label2id=label2id
    )

    print("\nA configurar Trainer...")
    args = TrainingArguments(
        output_dir=OUTPUT_MODEL_NAME,
        report_to="none",
        eval_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(label_list),
    )

    print(f"\nA treinar ({NUM_EPOCHS} épocas)...")
    trainer.train()

    print("\nA avaliar modelo final no conjunto de teste...")
    final_metrics = trainer.evaluate()
    print(json.dumps(final_metrics, indent=2))

    # Guardar métricas
    results_path = Path(OUTPUT_MODEL_NAME) / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(final_metrics, indent=2, ensure_ascii=False))
    print(f"\nMétricas guardadas em: {results_path}")
    print(f"Modelo guardado em:    {OUTPUT_MODEL_NAME}/")


if __name__ == "__main__":
    main()