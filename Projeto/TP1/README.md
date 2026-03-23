# Trabalho Prático 1 — Scripting no PLN (2025/26)

Seleção automática de frases representativas de textos usando modelos de linguagem baseados em **n-grams**, com análise de entidades nomeadas (NER) e geração de artigos em LaTeX.

---

## Estrutura do Projeto

```
.
├── textfiles/                  # Textos recolhidos (raw)
├── textfiles_clean/            # Textos limpos (prontos para processar)
├── ngram_results/
│   ├── frases_selecionadas.txt # 3 frases escolhidas por fonte
│   └── ner_entidades.txt       # Entidades nomeadas por fonte
├── latex_artigos/              # Ficheiros .tex gerados
│   ├── historia_futebol_inglaterra.tex
│   ├── primeira_liga.tex
│   └── revista_militar_futebol.tex
├── collect_and_clean.py        # Recolha das fontes (web + PDF)
├── clean_texts.py              # Limpeza e normalização dos textos
├── ngram.py                    # Modelo de n-grams e seleção de frases
├── ner.py                      # Análise de entidades nomeadas (spaCy)
└── generate_latex.py           # Geração dos artigos LaTeX
```

---

## Fontes Utilizadas

| Fonte | Tipo | URL |
|-------|------|-----|
| História do Futebol na Inglaterra | Página web (Wikipedia) | [link](https://pt.wikipedia.org/wiki/História_do_futebol_na_Inglaterra) |
| Primeira Liga | Página web (Wikipedia) | [link](https://pt.wikipedia.org/wiki/Primeira_Liga) |
| Revista Militar – Geopolítica e Futebol | PDF | [link](https://www.revistamilitar.pt/artigopdf/90) |

---

## Pipeline

### 1. Recolha das Fontes — `collect_and_clean.py`

- **Páginas web**: scraping com `requests` + `BeautifulSoup`, extraindo apenas parágrafos e cabeçalhos (`<p>`, `<h2>`, etc.), removendo infoboxes, navboxes, referências e outros elementos não-textuais da Wikipedia.
- **PDFs**: download do ficheiro e conversão para texto com `pdftotext -layout`.
- Os textos são guardados em `textfiles/`.

### 2. Limpeza dos Textos — `clean_texts.py`

Aplicada em duas camadas:

**Limpeza genérica** (`clean_common`), aplicada a todas as fontes:
- Normalização Unicode (NFC)
- Normalização de dashes (`–`, `—`, `−` → `-`)
- Remoção de espaços redundantes e linhas em excesso

**Limpeza específica por fonte:**
- *Wikipedia*: remoção de referências `[1]`, correção de espaços antes de pontuação, normalização de aspas
- *Revista Militar (PDF)*: remoção de cabeçalhos/rodapés de página, junção de palavras hifenizadas no fim de linha, remoção de títulos de secção residuais, correção de erros de extração do PDF (e.g. `"América do ul"` → `"América do Sul"`)

Os textos limpos são guardados em `textfiles_clean/`.

### 3. Modelo de N-grams e Seleção de Frases — `ngram.py`

**Tokenização:**
- Divisão em frases por pontuação final seguida de maiúscula
- Tokenização de palavras por expressão regular (apenas caracteres alfabéticos, incluindo acentuados)

**Construção do modelo:**
- Trigramas (N=3) com marcadores de início (`<s>`) e fim (`</s>`) de frase
- Contagem de n-grams e contextos (bigramas) para cálculo de probabilidades condicionais

**Scoring:**
- Cada frase é pontuada pela **log-probabilidade média** dos seus trigramas
- Aplica-se **suavização de Laplace** (add-1) para evitar probabilidades nulas
- As stopwords e tokens com menos de 3 caracteres são excluídos do cálculo
- São selecionadas as **3 frases com score mais alto**, com mínimo de 8 palavras

### 4. Análise de Entidades Nomeadas — `ner.py`

- Utiliza o modelo **`pt_core_news_sm`** do spaCy (português)
- As entidades são agrupadas por tipo e traduzidas para português:

| Tipo spaCy | Tipo exibido |
|------------|-------------|
| `PER` | Pessoa |
| `ORG` | Organização |
| `LOC` | Local |
| `GPE` | Local geopolítico |
| `MISC` | Miscelânea |
| `DATE` | Data |
| `EVENT` | Evento |

- Os resultados são guardados em `ngram_results/ner_entidades.txt`

### 5. Geração dos Artigos LaTeX — `generate_latex.py`

- Lê as frases selecionadas (`frases_selecionadas.txt`) e as entidades (`ner_entidades.txt`)
- Gera um ficheiro `.tex` por fonte com:
  - As 3 frases selecionadas no **resumo** (`abstract`)
  - O texto completo no corpo do artigo
  - As entidades nomeadas **destacadas a amarelo** com `\hl{}` (pacote `soul`)
  - Referência bibliográfica à fonte original
- O escape de caracteres especiais LaTeX é feito antes do highlight das entidades, com uso de placeholders para evitar highlights aninhados

---

## Como Executar

### Dependências

```bash
pip install requests beautifulsoup4 spacy
python -m spacy download pt_core_news_sm
```

É também necessário ter o `pdftotext` instalado (parte do pacote `poppler-utils`):

```bash
# Ubuntu/Debian
sudo apt install poppler-utils

### Execução passo a passo

```bash
# 1. Recolher os textos das fontes
python collect_and_clean.py

# 2. Limpar e normalizar os textos
python clean_texts.py

# 3. Construir o modelo de n-grams e selecionar frases
python ngram.py

# 4. Correr a análise NER
python ner.py

# 5. Gerar os artigos LaTeX
python generate_latex.py

# 6. Compilar os PDFs
cd latex_artigos
pdflatex historia_futebol_inglaterra.tex
pdflatex primeira_liga.tex
pdflatex revista_militar_futebol.tex
```

---

## Resultados

Cada PDF gerado contém:
- **Resumo**: as 3 frases mais representativas do texto, selecionadas pelo modelo de trigramas
- **Corpo**: o texto completo da fonte, com as entidades nomeadas destacadas a amarelo
- **Referências**: citação da fonte original