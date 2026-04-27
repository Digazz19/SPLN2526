"""
2_init_config.py
================
Gera o ficheiro de configuração do spaCy para treino NER em português.

Equivalente a:
    spacy init config configs/config.cfg --lang pt --pipeline ner --optimize accuracy --gpu

--optimize accuracy  -> arquitectura mais precisa (CNN com vetores pré-treinados)
--optimize efficiency -> mais rápida e leve, mas potencialmente menos precisa
"""

from pathlib import Path

from spacy.cli.init_config import fill_config, init_config

CONFIG_DIR = Path("configs")
BASE_CONFIG = CONFIG_DIR / "base_config.cfg"
FINAL_CONFIG = CONFIG_DIR / "config.cfg"


def main():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    print("A gerar base_config.cfg...")
    config = init_config(
        lang="pt",
        pipeline=["ner"],
        optimize="accuracy",
        gpu=True,
        pretraining=False,
        silent=False,
    )
    config.to_disk(BASE_CONFIG)

    print("A expandir para config.cfg final...")
    fill_config(output_file=FINAL_CONFIG, base_path=BASE_CONFIG)

    print(f"\nFeito. Configuração final: {FINAL_CONFIG}")


if __name__ == "__main__":
    main()