# TPC6 • Recuperação de Informação com TF-IDF

**Autor:** Diogo (PG55936)
**Data:** 27/04/2026

---

## Resumo

Este trabalho teve como objetivo a implementação de um sistema de **Recuperação
de Informação (IR)** baseado na métrica **TF-IDF** (*Term Frequency – Inverse
Document Frequency*), aplicada a um corpus de documentos em língua inglesa.

O processamento foi feito recorrendo às bibliotecas `nltk`, `collections` e
`math`, **sem uso de frameworks de IR externas**. A tokenização utilizou o
`word_tokenize` do NLTK com remoção de *stopwords* em inglês.

## Abordagem

A abordagem consistiu em quatro etapas principais:

1. **Tokenização** do corpus com remoção de palavras funcionais (*stopwords*).
2. **Cálculo do TF** (*Term Frequency*) por documento, normalizado pelo número
   de tokens.
3. **Cálculo do IDF** (*Inverse Document Frequency*) com base logarítmica
   (`log10(N/df)`), penalizando termos muito frequentes no corpus.
4. **Construção da matriz TF-IDF** e respetiva vetorização, permitindo
   representar cada documento como um vetor numérico no espaço do vocabulário.

A recuperação de informação foi implementada através de **similaridade cosseno**
entre o vetor da query e os vetores dos documentos, devolvendo os resultados
ordenados por relevância decrescente.

## Estrutura do Projeto

```
TPC6/
├── README.md
├── requirements.txt
└── tpc6.py             # Pipeline completo
```

## Como executar

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Descarregar dados do NLTK (uma única vez)
python3 -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"

# 3. Correr
python3 tpc6.py
```

## Corpus e query de teste

```python
CORPUS = [
    "the sky is blue",
    "the sun is bright",
    "the sun in the sky",
]
QUERY = "The bright sun"
```

## Resultados

### Pipeline de processamento

**1. Tokenização** (com remoção de *stopwords* em inglês — "the", "is", "in"):

```
[['sky', 'blue'], ['sun', 'bright'], ['sun', 'sky']]
```

**2. TF do primeiro documento:**

```
Counter({'sky': 0.5, 'blue': 0.5})
```

Cada termo ocorre uma vez num documento de 2 tokens, logo TF = 1/2 = 0.5.

**3. IDF do corpus:**

| Termo    |   IDF   | Documentos onde aparece (df) |
| :------- | :-----: | :--------------------------: |
| `sky`    | 0.1761  |              2               |
| `sun`    | 0.1761  |              2               |
| `blue`   | 0.4771  |              1               |
| `bright` | 0.4771  |              1               |

Termos que aparecem em mais documentos têm IDF mais baixo (são menos
discriminativos), conforme esperado.

**4. Matriz TF-IDF:**

| Doc | sky    | sun    | blue   | bright |
| :-: | :----: | :----: | :----: | :----: |
|  0  | 0.0880 |   0    | 0.2386 |   0    |
|  1  |   0    | 0.0880 |   0    | 0.2386 |
|  2  | 0.0880 | 0.0880 |   0    |   0    |

### Recuperação para a query `"The bright sun"`

```
1. [score=1.0000] Doc 1: "the sun is bright"
2. [score=0.2448] Doc 2: "the sun in the sky"
3. [score=0.0000] Doc 0: "the sky is blue"
```

## Discussão

- **Doc 1 obteve `score = 1.0000`** porque contém *exatamente* os mesmos
  termos relevantes da query (`sun`, `bright`) e nenhum outro — os vetores
  ficam colineares, pelo que a similaridade cosseno é máxima.

- **Doc 2 obteve `score = 0.2448`** porque partilha apenas `sun` com a query.
  O termo `sun` tem IDF baixo (aparece em 2 dos 3 documentos), o que limita o
  contributo para a similaridade.

- **Doc 0 obteve `score = 0.0000`** porque não partilha nenhum termo com a
  query (após remoção de *stopwords*) — `sky` e `blue` não aparecem em
  `"The bright sun"`.

- A **ordenação produzida é semanticamente correta**: o documento mais
  relevante para "The bright sun" é claramente "the sun is bright",
  seguido de "the sun in the sky", e por último "the sky is blue".