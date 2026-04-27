"""
1_convert.py
============
Converte os ficheiros IOB (treino e teste) para o formato binário do spaCy
(.spacy), usando o utilitário oficial `spacy convert`.

Equivalente a:
    spacy convert -c iob -n 10 arquivo_ner_train.iob ./datasets
    spacy convert -c iob -n 10 arquivo_ner_test.iob  ./datasets


A flag `-n 10` agrupa cada 10 frases num único Doc — o spaCy recomenda isto
para gerar dados de treino mais eficientes (menos overhead por documento).
"""

from pathlib import Path
from spacy.cli.convert import convert
from spacy.training.converters.conllu_to_docs import conllu_to_docs  # noqa: F401

DATASETS_DIR = Path("datasets")
INPUT_FILES = ["arquivo_ner_train.iob", "arquivo_ner_test.iob"]
SENTENCES_PER_DOC = 10  # Agrupa 10 frases por documento (recomendado para treino)


def main():
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    for iob_file in INPUT_FILES:
        if not Path(iob_file).exists():
            raise FileNotFoundError(f"Não encontrei o ficheiro: {iob_file}")

        print(f"\nConvert: {iob_file} -> {DATASETS_DIR}/")
        convert(
            input_path=Path(iob_file),
            output_dir=DATASETS_DIR,
            file_type="spacy",
            converter="ner",  # token-per-line; ver nota no docstring
            n_sents=SENTENCES_PER_DOC,
            lang="pt",
            silent=False,
        )

    print("\nFicheiros gerados:")
    for f in sorted(DATASETS_DIR.glob("*.spacy")):
        print(f"  {f}")


if __name__ == "__main__":
    main()