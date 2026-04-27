"""
4_evaluate.py
=============
Avalia o modelo spaCy treinado (output/model-best) no conjunto de teste,
e imprime as métricas (Precision, Recall, F1) globais e por tipo de entidade.

Equivalente a:
    spacy evaluate ./output/model-best ./datasets/arquivo_ner_test.spacy

Comparação com o modelo da aula (BERT) deve ser feita manualmente,
preenchendo a tabela no README.md com os valores deste script.
"""

import json
from pathlib import Path

from spacy.cli.evaluate import evaluate

MODEL_PATH = "output/model-best"
TEST_DATA = Path("datasets/arquivo_ner_test.spacy")
RESULTS_DIR = Path("results")
GPU_ID = 0


def main():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} não existe — corre primeiro 3_train.py"
        )
    if not TEST_DATA.exists():
        raise FileNotFoundError(
            f"{TEST_DATA} não existe — corre primeiro 1_convert.py"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_json = RESULTS_DIR / "spacy_metrics.json"

    print(f"A avaliar {MODEL_PATH} em {TEST_DATA}...")
    metrics = evaluate(
        model=MODEL_PATH,
        data_path=TEST_DATA,
        output=output_json,
        use_gpu=GPU_ID,
        silent=False,
    )

    # Métricas globais
    p = metrics.get("ents_p", 0.0)
    r = metrics.get("ents_r", 0.0)
    f = metrics.get("ents_f", 0.0)

    print("\n" + "=" * 50)
    print("Resultados spaCy (no conjunto de teste)")
    print("=" * 50)
    print(f"  Precision: {p:.4f}")
    print(f"  Recall:    {r:.4f}")
    print(f"  F1:        {f:.4f}")

    # Métricas por tipo de entidade
    per_type = metrics.get("ents_per_type", {})
    if per_type:
        print("\nPor tipo de entidade:")
        print(f"  {'Tipo':<15} {'Prec':>8} {'Rec':>8} {'F1':>8}")
        for label, m in sorted(per_type.items()):
            print(f"  {label:<15} {m['p']:>8.4f} {m['r']:>8.4f} {m['f']:>8.4f}")

    # Comparação com o modelo da aula
    print("\n" + "=" * 50)
    print("Comparação com modelo da aula (BERT)")
    print("=" * 50)
    print(f"  {'Modelo':<20} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print(f"  {'BERT (aula, ep.2)':<20} {0.9391:>8.4f} {0.9668:>8.4f} {0.9527:>8.4f}")
    print(f"  {'spaCy (treinado)':<20} {p:>8.4f} {r:>8.4f} {f:>8.4f}")

    diff = f - 0.9527
    if diff > 0:
        print(f"\n  -> spaCy supera o BERT em {diff:+.4f} F1")
    else:
        print(f"\n  -> BERT supera o spaCy em {-diff:.4f} F1")

    print(f"\nMétricas detalhadas guardadas em: {output_json}")


if __name__ == "__main__":
    main()