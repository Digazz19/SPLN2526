"""
corpus_builder.py
-----------------
Recolhe artigos da Wikipedia em inglês sobre futebol e guarda-os
num ficheiro JSON estruturado: corpus.json

Estrutura de cada documento:
{
    "id": int,
    "title": str,
    "text": str,          # texto limpo
    "url": str,
    "category": str       # ex: "club", "player", "competition", ...
}

Requisitos:
    pip install wikipedia-api spacy
    python -m spacy download en_core_web_sm
"""

import json
import re
import time
import wikipediaapi

# ---------------------------------------------------------------------------
# Títulos a recolher, organizados por categoria
# ---------------------------------------------------------------------------
ARTICLES = {
    "club": [
        "FC Barcelona", "Real Madrid CF", "Manchester United F.C.",
        "Liverpool F.C.", "Chelsea F.C.", "Arsenal F.C.",
        "Manchester City F.C.", "Tottenham Hotspur F.C.",
        "Bayern Munich", "Borussia Dortmund", "Juventus F.C.",
        "AC Milan", "Inter Milan", "AS Roma", "SSC Napoli",
        "Paris Saint-Germain F.C.", "Olympique de Marseille",
        "AFC Ajax", "S.L. Benfica", "FC Porto",
        "Sporting CP", "Celtic F.C.", "Rangers F.C.",
        "Club Atlético de Madrid", "Sevilla FC",
        "Valencia CF", "Real Sociedad",
        "Bayer 04 Leverkusen", "RB Leipzig",
        "Olympique Lyonnais", "Feyenoord",
        "Galatasaray S.K. (football)", "Fenerbahçe S.K. (football)",
        "Club América", "Boca Juniors", "Club Atlético River Plate",
        "Flamengo", "São Paulo FC", "Santos FC",
        "Club de Fútbol América",
    ],
    "player": [
        "Lionel Messi", "Cristiano Ronaldo", "Pelé",
        "Diego Maradona", "Ronaldo (Brazilian footballer)",
        "Zinedine Zidane", "Ronaldinho", "Thierry Henry",
        "David Beckham", "Zlatan Ibrahimović",
        "Neymar", "Kylian Mbappé", "Erling Haaland",
        "Luka Modrić", "Kevin De Bruyne",
        "Mohamed Salah", "Robert Lewandowski",
        "Virgil van Dijk", "Harry Kane",
        "Sadio Mané", "Karim Benzema",
        "Paolo Maldini", "Gianluigi Buffon",
        "Xavi Hernández", "Andrés Iniesta",
        "Franck Ribéry", "Arjen Robben",
        "Wayne Rooney", "Steven Gerrard",
        "Frank Lampard", "Didier Drogba",
    ],
    "competition": [
        "UEFA Champions League", "UEFA Europa League",
        "FIFA World Cup", "UEFA European Championship",
        "Copa América", "Premier League",
        "La Liga", "Serie A", "Bundesliga",
        "Ligue 1", "Primeira Liga",
        "FIFA Club World Cup", "UEFA Super Cup",
        "Copa del Rey", "DFB-Pokal", "FA Cup",
        "EFL Cup", "Coppa Italia",
        "AFC Champions League", "CONMEBOL Libertadores",
    ],
    "stadium": [
        "Camp Nou", "Santiago Bernabéu Stadium",
        "Old Trafford", "Anfield",
        "Wembley Stadium", "Allianz Arena",
        "Signal Iduna Park", "San Siro",
        "Stadio Olimpico", "Parc des Princes",
        "Estádio da Luz", "Estádio do Dragão",
        "Emirates Stadium", "Etihad Stadium"
    ],
    "concept": [
        "Association football", "Football (word)",
        "History of association football",
        "Offside (association football)",
        "Penalty kick (association football)",
        "VAR (association football)",
        "Transfer (association football)",
        "Football hooliganism",
        "FIFA", "UEFA",
        "Premier League",
        "Football pitch",
        "Goalkeeper (association football)",
        "FIFA World Rankings",
    ],
}

# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Remove linhas muito curtas, referências estilo wiki e espaços extras."""
    # Remove conteúdo entre == (cabeçalhos) que ficam como texto solto
    text = re.sub(r"={2,}[^=]+=={2,}", "", text)
    # Remove referências numéricas [1], [2], ...
    text = re.sub(r"\[\d+\]", "", text)
    # Colapsa múltiplos espaços/newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def fetch_article(wiki, title: str, category: str) -> dict | None:
    """Obtém e limpa um artigo da Wikipedia. Retorna None se não existir."""
    page = wiki.page(title)
    if not page.exists():
        print(f"  [SKIP] '{title}' não encontrado.")
        return None

    text = clean_text(page.text)
    if len(text.split()) < 100:          # ignora artigos muito curtos
        print(f"  [SKIP] '{title}' demasiado curto.")
        return None

    print(f"  [OK]   '{title}' ({len(text.split())} palavras)")
    return {
        "title": title,
        "text": text,
        "url": page.fullurl,
        "category": category,
    }


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def build_corpus(output_path: str = "corpus.json") -> list[dict]:
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="SPLN-TP2-FootballCorpus/1.0 (university project)",
    )

    corpus = []
    doc_id = 0

    for category, titles in ARTICLES.items():
        print(f"\n=== Categoria: {category.upper()} ===")
        for title in titles:
            doc = fetch_article(wiki, title, category)
            if doc:
                doc["id"] = doc_id
                corpus.append(doc)
                doc_id += 1
            time.sleep(0.3)   # respeita rate limit da Wikipedia

    print(f"\nCorpus construído: {len(corpus)} documentos")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"Guardado em '{output_path}'")
    return corpus


# ---------------------------------------------------------------------------
# Análise rápida do corpus
# ---------------------------------------------------------------------------

def corpus_stats(corpus: list[dict]) -> None:
    from collections import Counter

    print("\nEstatísticas do corpus:")
    print(f"   Total de documentos : {len(corpus)}")

    cat_counts = Counter(d["category"] for d in corpus)
    for cat, count in sorted(cat_counts.items()):
        print(f"   {cat:<15}: {count} documentos")

    total_words = sum(len(d["text"].split()) for d in corpus)
    avg_words = total_words // len(corpus)
    print(f"   Total de palavras   : {total_words:,}")
    print(f"   Média por documento : {avg_words:,} palavras")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    corpus = build_corpus("corpus.json")
    corpus_stats(corpus)