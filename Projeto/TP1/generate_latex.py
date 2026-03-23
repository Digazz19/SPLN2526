import os
import re

INPUT_DIR   = "textfiles_clean"
NER_FILE    = "ngram_results/ner_entidades.txt"
FRASES_FILE = "ngram_results/frases_selecionadas.txt"
OUTPUT_DIR  = "latex_artigos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SOURCES = [
    {
        "label":    "História do Futebol na Inglaterra",
        "filename": os.path.join(INPUT_DIR, "historia_futebol_inglaterra_clean.txt"),
        "bibkey":   "wikipedia_inglaterra",
        "bibtitle": "História do futebol na Inglaterra",
        "biburl":   "https://pt.wikipedia.org/wiki/Hist\\%C3\\%B3ria\\_do\\_futebol\\_na\\_Inglaterra",
        "outfile":  "historia_futebol_inglaterra.tex",
    },
    {
        "label":    "Primeira Liga",
        "filename": os.path.join(INPUT_DIR, "primeira_liga_clean.txt"),
        "bibkey":   "wikipedia_primeira_liga",
        "bibtitle": "Primeira Liga",
        "biburl":   "https://pt.wikipedia.org/wiki/Primeira\\_Liga",
        "outfile":  "primeira_liga.tex",
    },
    {
        "label":    "Revista Militar – Geopolítica e Futebol",
        "filename": os.path.join(INPUT_DIR, "revista_militar_futebol_clean.txt"),
        "bibkey":   "revista_militar",
        "bibtitle": "Geopolítica e o Desporto de Massas --- O Futebol",
        "biburl":   "https://www.revistamilitar.pt/artigopdf/90",
        "outfile":  "revista_militar_futebol.tex",
    },
]

# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_frases(path: str) -> dict:
    """Lê frases_selecionadas.txt → {label: [frase1, frase2, frase3]}."""
    result = {}
    with open(path, encoding="utf-8") as f:
        content = f.read()
    blocks = re.split(r'={10,}', content)
    i = 1
    while i < len(blocks) - 1:
        label        = blocks[i].strip()
        frases_block = blocks[i + 1]
        frases = re.findall(r'^\[\d+\]\s*(.+)', frases_block, re.MULTILINE)
        if label and frases:
            result[label] = frases
        i += 2
    return result

def parse_ner(path: str) -> dict:
    """Lê ner_entidades.txt → {label: {tipo: [entidade, ...]}}."""
    result = {}
    current_source = None
    current_type   = None
    sep_count      = 0

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if "=" * 10 in line:
            sep_count += 1
            if sep_count % 2 == 1:
                i += 1
                while i < len(lines) and not lines[i].strip():
                    i += 1
                if i < len(lines):
                    current_source = lines[i].strip()
                    result[current_source] = {}
                    current_type = None
            i += 1
            continue
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current_type = stripped[1:-1]
            if current_source and current_type not in result[current_source]:
                result[current_source][current_type] = []
        elif stripped.startswith("- ") and current_type and current_source:
            ent = stripped[2:].strip()
            if len(ent) > 1 and ent not in ('"', "\u201c", "\u201d"):
                result[current_source][current_type].append(ent)
        i += 1
    return result

# ── Escape LaTeX ─────────────────────────────────────────────────────────────

def latex_escape(text: str) -> str:
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&",  "\\&"),
        ("%",  "\\%"),
        ("$",  "\\$"),
        ("#",  "\\#"),
        ("_",  "\\_"),
        ("{",  "\\{"),
        ("}",  "\\}"),
        ("~",  "\\textasciitilde{}"),
        ("^",  "\\textasciicircum{}"),
        ("–",  "--"),
        ("—",  "---"),
        (""",  "``"),
        (""",  "''"),
        ("«",  "``"),
        ("»",  "''"),
    ]
    for char, escaped in replacements:
        text = text.replace(char, escaped)
    return text

# ── Highlight de entidades no texto ──────────────────────────────────────────

def highlight_entities(text: str, ner: dict) -> str:
    """
    Envolve as ocorrências das entidades em \\hl{...}.
    Recebe texto já escapado para LaTeX.
    Usa placeholders para evitar highlights aninhados.
    """
    all_entities = []
    for ents in ner.values():
        all_entities.extend(ents)
    # Ordenar por comprimento decrescente - entidades maiores têm prioridade
    all_entities = sorted(set(all_entities), key=len, reverse=True)

    # Mapa de placeholder -> comando hl final
    placeholders = {}
    counter = [0]

    for ent in all_entities:
        if len(ent) < 4:
            continue
        ent_tex = latex_escape(ent)
        pattern = re.escape(ent_tex)
        try:
            if re.search(pattern, text):
                placeholder = f"HLPLACEHOLDER{counter[0]}END"
                counter[0] += 1
                placeholders[placeholder] = f"\\hl{{{ent_tex}}}"
                text = re.sub(pattern, placeholder, text)
        except re.error:
            continue

    # Substituir placeholders pelos comandos \hl reais
    for placeholder, hl_cmd in placeholders.items():
        text = text.replace(placeholder, hl_cmd)

    return text

# ── Geração do artigo LaTeX ──────────────────────────────────────────────────

def make_article(source: dict, frases: list, ner: dict) -> str:
    title    = latex_escape(source["label"])
    bibkey   = source["bibkey"]
    bibtitle = latex_escape(source["bibtitle"])
    biburl   = source["biburl"]

    # Escapar o texto para LaTeX primeiro, depois fazer highlight das entidades
    with open(source["filename"], encoding="utf-8") as f:
        raw_text = f.read()

    body_text = latex_escape(raw_text)
    body_text = highlight_entities(body_text, ner)

    frases_items = "\n".join(
        f"  \\item {latex_escape(fr)}" for fr in frases
    )

    return rf"""
\documentclass[a4paper,12pt]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage[portuguese]{{babel}}
\usepackage{{geometry}}
\usepackage[dvipsnames]{{xcolor}}
\usepackage{{soul}}
\usepackage{{hyperref}}
\usepackage{{microtype}}
\geometry{{margin=2.5cm}}

% \hl usa soul que permite quebra de linha dentro do highlight
\sethlcolor{{yellow}}

\title{{{title}}}
\author{{Trabalho Prático 1 --- Scripting no PLN}}
\date{{2025/26}}

\begin{{document}}

\maketitle

\begin{{abstract}}
As seguintes frases foram selecionadas através de um modelo de linguagem
baseado em \textit{{n-grams}} (trigramas) com \textit{{scoring}} por
log-probabilidade média e suavização de Laplace:
\begin{{itemize}}
{frases_items}
\end{{itemize}}
\end{{abstract}}

\section{{Texto da Fonte}}

{body_text}

\bibliographystyle{{plain}}
\begin{{thebibliography}}{{1}}
  \bibitem{{{bibkey}}}
  {bibtitle}.
  Disponível em: \url{{{biburl}}}.
  Acedido em março de 2026.
\end{{thebibliography}}

\end{{document}}
""".lstrip()

# ── Pipeline principal ───────────────────────────────────────────────────────

def main():
    frases_dict = parse_frases(FRASES_FILE)
    ner_dict    = parse_ner(NER_FILE)

    for source in SOURCES:
        label  = source["label"]
        frases = frases_dict.get(label, [])
        ner    = ner_dict.get(label, {})

        print(f"A gerar: {source['outfile']}")
        tex = make_article(source, frases, ner)

        out_path = os.path.join(OUTPUT_DIR, source["outfile"])
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(tex)
        print(f"  Guardado em: {out_path}")

    print("\nPara compilar os PDFs:")
    print(f"  cd {OUTPUT_DIR}")
    print("  pdflatex historia_futebol_inglaterra.tex")
    print("  pdflatex primeira_liga.tex")
    print("  pdflatex revista_militar_futebol.tex")

if __name__ == "__main__":
    main()