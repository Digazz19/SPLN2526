import requests
import subprocess
import re
import unicodedata
import os
from bs4 import BeautifulSoup

# ── Configuração ────────────────────────────────────────────────────────────
OUTPUT_DIR = "textfiles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}

WEB_SOURCES = [
    {
        "url": "https://pt.wikipedia.org/wiki/História_do_futebol_na_Inglaterra",
        "filename": os.path.join(OUTPUT_DIR, "historia_futebol_inglaterra.txt"),
        "label": "História do Futebol na Inglaterra (Wikipedia)",
    },
    {
        "url": "https://pt.wikipedia.org/wiki/Primeira_Liga",
        "filename": os.path.join(OUTPUT_DIR, "primeira_liga.txt"),
        "label": "Primeira Liga (Wikipedia)",
    },
]

PDF_SOURCES = [
    {
        "url": "https://www.revistamilitar.pt/artigopdf/90",
        "filename": os.path.join(OUTPUT_DIR, "revista_militar_futebol.txt"),
        "label": "Revista Militar – artigo PDF",
    },
]


# ── Funções de limpeza ───────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalização unicode, remoção de espaços redundantes e linhas vazias."""
    text = unicodedata.normalize("NFC", text)

    text = re.sub(r'[^\S\n\t ]+', ' ', text)

    text = re.sub(r'\n{3,}', '\n\n', text)

    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def scrape_wikipedia(url: str) -> str:
    """Extrai texto útil de uma página Wikipedia em português."""
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remover elementos que não são conteúdo principal
    for tag in soup.select(
        ".infobox, .navbox, .reflist, .references, "
        ".mw-editsection, sup.reference, .toc, "
        "table.wikitable, .thumb, .gallery, "
        ".hatnote, .sistersitebox, .noprint"
    ):
        tag.decompose()

    content_div = soup.find(id="mw-content-text")
    if not content_div:
        raise ValueError(f"Não foi encontrado conteúdo em: {url}")

    # Extrair apenas parágrafos e cabeçalhos (ignora listas de refs, etc.)
    parts = []
    for elem in content_div.find_all(["h2", "h3", "h4", "p"]):
        text = elem.get_text(separator=" ")
        text = text.strip()
        if text and len(text) > 20: 
            parts.append(text)

    return clean_text("\n\n".join(parts))


def download_and_extract_pdf(url: str) -> str:
    """Descarrega um PDF de uma URL, converte com pdftotext e extrai o texto."""
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    tmp_pdf = os.path.join(OUTPUT_DIR, "_tmp_download.pdf")
    tmp_txt = os.path.join(OUTPUT_DIR, "_tmp_download.txt")

    with open(tmp_pdf, "wb") as f:
        f.write(response.content)

    text = extract_pdf_text(tmp_pdf, tmp_txt)

    os.remove(tmp_pdf)
    if os.path.exists(tmp_txt):
        os.remove(tmp_txt)

    return text


def extract_pdf_text(pdf_path: str, txt_path: str) -> str:
    """Usa pdftotext para converter o PDF em .txt e lê o resultado."""
    result = subprocess.run(
        ["pdftotext", "-layout", pdf_path, txt_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext falhou: {result.stderr.strip()}")

    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    return clean_text(raw_text)


# ── Pipeline principal ───────────────────────────────────────────────────────

def collect_all():
    results = {}

    # Páginas web (Wikipedia)
    for source in WEB_SOURCES:
        print(f"\n[WEB] A recolher: {source['label']}")
        try:
            text = scrape_wikipedia(source["url"])
            with open(source["filename"], "w", encoding="utf-8") as f:
                f.write(text)
            results[source["label"]] = {
                "file": source["filename"],
                "chars": len(text),
            }
            print(f"Guardado em '{source['filename']}' ({len(text):,} caracteres)")
        except Exception as e:
            print(f"Erro: {e}")

    # PDFs remotos
    for source in PDF_SOURCES:
        print(f"\n[PDF] A recolher: {source['label']}")
        try:
            text = download_and_extract_pdf(source["url"])
            with open(source["filename"], "w", encoding="utf-8") as f:
                f.write(text)
            results[source["label"]] = {
                "file": source["filename"],
                "chars": len(text),
            }
            print(f"Guardado em '{source['filename']}' ({len(text):,} caracteres)")
        except Exception as e:
            print(f"Erro: {e}")

    print("\n" + "═" * 60)
    print("RESUMO DA RECOLHA")
    print("═" * 60)
    for label, info in results.items():
        print(f"\n{label}")
        print(f"  Ficheiro : {info['file']}")
        print(f"  Tamanho  : {info['chars']:,} caracteres")

    return results


if __name__ == "__main__":
    collect_all()