import os
import spacy
from collections import defaultdict

INPUT_DIR  = "textfiles_clean"
OUTPUT_DIR = "ngram_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SOURCES = [
    {
        "filename": os.path.join(INPUT_DIR, "historia_futebol_inglaterra_clean.txt"),
        "label":    "História do Futebol na Inglaterra",
    },
    {
        "filename": os.path.join(INPUT_DIR, "primeira_liga_clean.txt"),
        "label":    "Primeira Liga",
    },
    {
        "filename": os.path.join(INPUT_DIR, "revista_militar_futebol_clean.txt"),
        "label":    "Revista Militar – Geopolítica e Futebol",
    },
]

# Mapeamento dos tipos de entidade do spaCy para português
LABEL_PT = {
    "PER":   "Pessoa",
    "ORG":   "Organização",
    "LOC":   "Local",
    "GPE":   "Local geopolítico",
    "MISC":  "Miscelânea",
    "DATE":  "Data",
    "EVENT": "Evento",
}

# ── NER ──────────────────────────────────────────────────────────────────────

def run_ner(text: str, nlp) -> dict:
    """
    Corre o NER do spaCy sobre o texto e devolve as entidades agrupadas
    por tipo, sem repetições.
    """
    doc = nlp(text)
    entities = defaultdict(set)
    for ent in doc.ents:
        label = LABEL_PT.get(ent.label_, ent.label_)
        entities[label].add(ent.text.strip())
    # Converter sets para listas ordenadas
    return {label: sorted(ents) for label, ents in sorted(entities.items())}

def print_entities(entities: dict):
    for label, ents in entities.items():
        print(f"\n  [{label}]")
        for e in ents:
            print(f"    - {e}")

def save_entities(f, label: str, entities: dict):
    f.write(f"\n{'='*60}\n")
    f.write(f"{label}\n")
    f.write(f"{'='*60}\n")
    for etype, ents in entities.items():
        f.write(f"\n  [{etype}]\n")
        for e in ents:
            f.write(f"    - {e}\n")

# ── Pipeline principal ───────────────────────────────────────────────────────

def main():
    print("A carregar modelo spaCy (pt_core_news_sm)...")
    nlp = spacy.load("pt_core_news_sm")
    print("Modelo carregado.\n")

    all_results = []

    for source in SOURCES:
        print(f"\n{'='*60}")
        print(f"Fonte: {source['label']}")
        print(f"{'='*60}")

        with open(source["filename"], "r", encoding="utf-8") as f:
            text = f.read()

        entities = run_ner(text, nlp)
        print_entities(entities)

        all_results.append({
            "label":    source["label"],
            "entities": entities,
        })

    # Guardar resultados em ficheiro
    out_path = os.path.join(OUTPUT_DIR, "ner_entidades.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ENTIDADES NOMEADAS POR FONTE\n")
        for r in all_results:
            save_entities(f, r["label"], r["entities"])
    print(f"\n\nEntidades guardadas em: {out_path}")

if __name__ == "__main__":
    main()