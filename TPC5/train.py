"""
3_train.py
==========
Treina o modelo spaCy NER na GPU 0 (RTX 3060).

Equivalente a:
    spacy train configs/config.cfg \\
        --output ./output \\
        --paths.train ./datasets/arquivo_ner_train.spacy \\
        --paths.dev   ./datasets/arquivo_ner_test.spacy \\
        --gpu-id 0

Pré-requisitos:
  - datasets/arquivo_ner_{train,test}.spacy gerados por convert.py
  - configs/config.cfg gerado por initConfig.py
  - spacy[cuda13x] instalado (ou cuda11x, conforme drivers)
"""

from pathlib import Path

from spacy.cli.train import train

CONFIG_PATH = Path("configs/config.cfg")
OUTPUT_DIR = Path("output")
TRAIN_DATA = Path("datasets/arquivo_ner_train.spacy")
DEV_DATA = Path("datasets/arquivo_ner_test.spacy")
GPU_ID = 0  # RTX 3060


def main():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"{CONFIG_PATH} não existe — corre primeiro initConfig.py"
        )
    if not TRAIN_DATA.exists() or not DEV_DATA.exists():
        raise FileNotFoundError(
            f"{TRAIN_DATA} ou {DEV_DATA} não existem — "
            "corre primeiro convert.py"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"A treinar modelo spaCy NER na GPU {GPU_ID}...")
    print(f"  Treino: {TRAIN_DATA}")
    print(f"  Dev:    {DEV_DATA}")
    print(f"  Output: {OUTPUT_DIR}/")

    train(
        config_path=CONFIG_PATH,
        output_path=OUTPUT_DIR,
        use_gpu=GPU_ID,
        overrides={
            "paths.train": str(TRAIN_DATA),
            "paths.dev": str(DEV_DATA),
        },
    )

    print(f"\nTreino concluído. Modelo final em: {OUTPUT_DIR / 'model-best'}/")


if __name__ == "__main__":
    main()