import re
import unicodedata
import os

INPUT_DIR  = "textfiles"
OUTPUT_DIR = "textfiles_clean"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Limpeza genérica (aplicada a todas as fontes) ────────────────────────────

def clean_common(text: str) -> str:
    """Limpeza partilhada por todas as fontes."""
    text = unicodedata.normalize("NFC", text)

    text = re.sub(r'[–—−]', '-', text)

    text = re.sub(r'[^\S\n\t ]+', ' ', text)

    text = re.sub(r'[ \t]{2,}', ' ', text)

    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ── Limpeza específica: páginas web (Wikipedia) ──────────────────────────────

def clean_wikipedia(text: str, truncate_at: str = None) -> str:
    """Remove artefactos típicos do scraping de páginas Wikipedia.

    truncate_at: se fornecido, o texto é cortado no fim da linha que
                 contiver essa substring (inclusive).
    """
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[nota \d+\]', '', text)

    lines = text.splitlines()
    lines = [l for l in lines if len(l.strip()) >= 3]
    if truncate_at:
        for i, line in enumerate(lines):
            if truncate_at in line:
                lines = lines[:i + 1]
                break
    text = '\n'.join(lines)

    text = re.sub(r' ([.,;:!?])', r'\1', text)

    text = re.sub(r'(?<=\s)" ([^"]+?) "(?=[\s,.:;!?])', r'"\1"', text)

    text = re.sub(r'\( ([^)]+)\)', r'(\1)', text)
    text = re.sub(r'\(([^)]+) \)', r'(\1)', text)
    return clean_common(text)

# ── Limpeza específica: PDF da Revista Militar ───────────────────────────────

def clean_revista_militar(text: str) -> str:
    """
    Problemas deste PDF extraído com pdftotext -layout:
      1. Cabeçalhos/rodapés de página repetidos em cada página
      2. Indentação com espaços à esquerda (artefacto do -layout)
      3. Hifenização no fim de linha (palavras cortadas: 'globa-\nlização')
      3b. Linhas partidas a meio de parágrafo
      3c. Números de referência inline colados às palavras
      4. Notas de rodapé numeradas no fim do documento
      5. Linhas muito curtas / fragmentos de layout
      6. Legendas de imagens (Foto da ..., Abertura do ..., Fonte: ...)
      7. Cabeçalho do artigo (título, autor, epígrafe)
      8. Palavra cortada "América do ul" → "América do Sul"
      9. Referências inline soltas (ex: "mob football 2,")
    """
    text = re.sub(r'[ \t]*Revista Militar N\.º.*?\n', '', text)
    text = re.sub(r'[ \t]*:: Neste pdf - página \d+ de \d+ ::\n', '', text)

    lines = text.splitlines()
    lines = [l.lstrip() for l in lines]
    text = '\n'.join(lines)

    text = re.sub(r'\nFoto d[ao][^\n]+', '', text)
    text = re.sub(r'\nAbertura do[^\n]+', '', text)
    text = re.sub(r'\nFonte:[^\n]+', '', text)
    text = re.sub(r'Fonte:\s*www\.[^\s]+\s*', '', text)
    text = re.sub(r'\(Fonte:[^\)]+\)', '', text)
    text = re.sub(r'\n\d{4}\.\s*\n', '\n', text)
    text = re.sub(r'\n\d{4}\.\s*(?=[a-záéíóúA-ZÁÉÍÓÚ])', ' ', text)

    text = re.sub(r'^[^\n]+\n+Coronel\n[^\n]+\n', '', text)
    text = re.sub(r'Estudo de [^\n]+\n', '', text)

    text = re.sub(r'-\n(\S)', lambda m: m.group(1), text)

    text = re.sub(r'\bA\n(?=frica\b)', 'Á', text)
    text = re.sub(r'(?<![.!?:])\n(?=[a-záéíóúàâêôãõüçA-ZÁÉÍÓÚÀÂÊÔÃÕ])', ' ', text)

    text = re.sub(r'\bde frica\b', 'de África', text)

    text = re.sub(r'(?<=[a-záéíóúàâêôãõüç])(\d+)(?=[\s,.:;!?\"\')\-]|$)', '', text)

    text = re.sub(r'(?<=[a-záéíóúàâêôãõüç]) (\d{1,2}) (?=a |à |e |o |os |as |que |se |na |no |da |do )', r' ', text)

    text = re.sub(r'(de \d{1,2}),', r'NUMPLACEHOLDER\1NUMEND,', text)
    text = re.sub(r'(?<=[a-záéíóúàâêôãõüç]) (\d{1,2})(?=, )', '', text)
    text = text.replace('NUMPLACEHOLDER', '').replace('NUMEND', '')

    text = re.sub(r'\bIntrodução\b\s+(?=[A-ZÁÉÍÓÚ])', '', text)
    text = re.sub(r'\d+\.\s+[A-ZÁÉÍÓÚ][^\.\n]{5,40}\s+(?=[A-ZÁÉÍÓÚ][a-záéíóú])', '', text)
    text = re.sub(r'\b[a-d]\.\s+[A-ZÁÉÍÓÚ][^\.\n]{3,30}\s+(?=[A-ZÁÉÍÓÚ][a-záéíóú])', '', text)

    TITULOS_RESIDUAIS = [
        r'^\s*Internacionais no Futebol\s*$',
        r'^\s*Futebol\s*$',
        r'^\s*Política\s*$',
        r'^\s*Multipolar\s*$',
        r'^\s*Etnografia\s*$',
    ]
    lines = text.splitlines()
    lines = [l for l in lines if not any(re.match(p, l) for p in TITULOS_RESIDUAIS)]
    text = '\n'.join(lines)

    text = text.replace(
        'no Campeonato da Europa de Argentina,',
        'no Campeonato da Europa de 1960. Em 1978, por altura do Mundial na Argentina,'
    )
    text = text.replace(
        'para o Campeonato do Mundo de El Salvador acabou por vencer',
        'para o Campeonato do Mundo de 1970. Tendo perdido por 1 a 0 no desafio da 1ª mão, El Salvador acabou por vencer'
    )

    text = re.sub(r'"(\w+)"\d{1,2}(?=\s)', r'"\1"', text)

    text = re.sub(r'\n\d{1,2} [A-ZÁÉÍÓÚ][^\n]+', '', text)

    text = text.replace('América do ul', 'América do Sul')
    text = text.replace('com ma derrota', 'com uma derrota')
    text = text.replace('por ano United', 'por ano. Outro exemplo é o do Manchester United')

    lines = text.splitlines()
    lines = [l for l in lines if len(l.strip()) >= 4]

    for i, line in enumerate(lines):
        if line.strip().lower() == "bibliografia":
            lines = lines[:i]
            break

    text = '\n'.join(lines)
    return clean_common(text)

# ── Pipeline ─────────────────────────────────────────────────────────────────

sources = [
    {
        "input":    os.path.join(INPUT_DIR, "historia_futebol_inglaterra.txt"),
        "output":   os.path.join(OUTPUT_DIR, "historia_futebol_inglaterra_clean.txt"),
        "label":    "História do Futebol na Inglaterra",
        "clean_fn": lambda t: clean_wikipedia(t),
    },
    {
        "input":    os.path.join(INPUT_DIR, "primeira_liga.txt"),
        "output":   os.path.join(OUTPUT_DIR, "primeira_liga_clean.txt"),
        "label":    "Primeira Liga",
        "clean_fn": lambda t: clean_wikipedia(
            t,
            truncate_at="O acesso às competições de clubes da UEFA é feito tendo por base"
        ),
    },
    {
        "input":    os.path.join(INPUT_DIR, "revista_militar_futebol.txt"),
        "output":   os.path.join(OUTPUT_DIR, "revista_militar_futebol_clean.txt"),
        "label":    "Revista Militar – Geopolítica e Futebol",
        "clean_fn": clean_revista_militar,
    },
]

def main():
    for src in sources:
        print(f"\n[A limpar] {src['label']}")
        with open(src["input"], "r", encoding="utf-8") as f:
            raw = f.read()

        cleaned = src["clean_fn"](raw)

        with open(src["output"], "w", encoding="utf-8") as f:
            f.write(cleaned)

        reduction = (1 - len(cleaned) / len(raw)) * 100
        print(f"  Antes  : {len(raw):>7,} chars")
        print(f"  Depois : {len(cleaned):>7,} chars  ({reduction:.1f}% removido)")
        print(f"  Guardado em: {src['output']}")
        print(f"  Prévia :\n{cleaned[:300]}\n  ...")

if __name__ == "__main__":
    main()