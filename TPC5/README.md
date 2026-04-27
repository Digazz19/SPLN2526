# TPC5 • Treino e Comparação de Modelos NER (spaCy vs. BERT)

**Autor:** Diogo Abreu(PG55936)
**Data:** 19/04/2026

---

## Resumo

Treino de um modelo de **Reconhecimento de Entidades Mencionadas (NER)** com `spaCy`,
usando os ficheiros IOB fornecidos (`arquivo_ner_train.iob` e `arquivo_ner_test.iob`),
e comparação dos resultados com o modelo da aula (BERT).

## Entidades

O dataset contém 5 tipos de entidade:

- `Pessoa`
- `Local`
- `Organizacao`
- `Data`
- `Profissao`

## Estrutura do Projeto

```
TPC5/
├── README.md
├── requirements.txt            # Dependências para a parte do spaCy
├── requirementsAula.txt        # Dependências para a parte da aula (BERT)
├── arquivo_ner_train.iob       # Dados de treino (fornecidos)
├── arquivo_ner_test.iob        # Dados de teste (fornecidos)
├── aula9.py                    # Exercício da aula (BERT) em script Python
├── convert.py                  # IOB -> .spacy (via spacy convert)
├── initConfig.py               # Gera config.cfg (spacy init config)
├── train.py                    # Treina o modelo (spacy train)
├── evaluateSpacy.py                 # Avalia model-best em arquivo_ner_test.spacy
├── datasets/                   # .spacy gerados
├── configs/                    # config.cfg
├── output/                     # Modelo spaCy treinado (model-best, model-last)
└── my_model/                   # Modelo BERT da aula (checkpoints + métricas)
```

## Como executar

### Parte 1 — Exercício da aula (BERT, opcional)

```bash
pip install -r requirementsAula.txt
python aula9.py
```

### Parte 2 — Treino com spaCy nos ficheiros IOB do TPC

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Descarregar vectores pré-treinados em PT (necessários para --optimize accuracy)
python -m spacy download pt_core_news_lg

# 3. Converter IOB -> .spacy
python convert.py

# 4. Gerar config.cfg
python initConfig.py

# 5. Treinar (usa GPU 0 - RTX 3060)
python train.py

# 6. Avaliar o modelo treinado
python evaluateSpacy.py
```

## Comandos equivalentes em CLI (referência)

```bash
spacy convert -c iob -n 10 arquivo_ner_train.iob ./datasets
spacy convert -c iob -n 10 arquivo_ner_test.iob  ./datasets
spacy init config configs/config.cfg --lang pt --pipeline ner --optimize accuracy --gpu
spacy train configs/config.cfg --output ./output \
    --paths.train ./datasets/arquivo_ner_train.spacy \
    --paths.dev   ./datasets/arquivo_ner_test.spacy \
    --gpu-id 0
spacy evaluate ./output/model-best ./datasets/arquivo_ner_test.spacy
```

## Resultados da aula (BERT — referência)
 
| Epoch | Validation Loss | Precision | Recall |   F1   | Accuracy |
| :---: | :-------------: | :-------: | :----: | :----: | :------: |
|   1   |     0.0777      |  0.9312   | 0.9592 | 0.9450 |  0.9811  |
|   2   |     0.0683      |  0.9391   | 0.9668 | 0.9527 |  0.9838  |
 
Melhor resultado da aula: **F1 = 0.9527** (época 2).
 
## Resultados do spaCy
 
Modelo treinado durante 25 épocas (~9200 steps), com paragem por *early
stopping*. O melhor checkpoint (`model-best`) foi atingido por volta da
época 20.
 
### Métricas globais (no conjunto de teste)
 
| Métrica   |  Valor  |
| :-------- | :-----: |
| Precision | 0.9895  |
| Recall    | 0.9856  |
| F1        | 0.9875  |
 
### Métricas por tipo de entidade
 
|     Tipo     | Precision | Recall |   F1   |  Suporte (treino) |
| :----------- | :-------: | :----: | :----: | :---------------: |
| `Data`        |  1.0000   | 0.9936 | 0.9968 |   1858            |
| `Local`       |  0.9973   | 0.9938 | 0.9956 |   4244            |
| `Pessoa`      |  0.9906   | 0.9899 | 0.9903 |   5895            |
| `Organizacao` |  0.9118   | 0.9394 | 0.9254 |    397            |
| `Profissao`   |  0.9366   | 0.8867 | 0.9110 |    465            |
 
## Comparação com o modelo da aula
 
| Modelo                |  Prec  |   Rec  |   F1   |
| :-------------------- | :----: | :----: | :----: |
| BERT (aula, época 2)  | 0.9391 | 0.9668 | 0.9527 |
| **spaCy (treinado)**  | **0.9895** | **0.9856** | **0.9875** |
 
O modelo spaCy **supera o BERT da aula em +0.0348 pontos de F1**
(equivalente a uma redução de erro de ~73%, de 4.73% para 1.25%).
 
### Discussão
 
Os resultados mostram um padrão interessante:
 
- **Classes maioritárias e regulares** (`Data`, `Local`, `Pessoa`) atingem
  F1 ≥ 0.99. As datas, em particular, têm `Precision = 1.00` — o modelo
  aprende perfeitamente o seu padrão regular ("DD de Mês de AAAA",
  "AAAA-MM-DD", etc.). `Local` e `Pessoa` beneficiam de terem milhares de
  exemplos no treino, com nomes próprios facilmente reconhecíveis
  (capitalização, contexto sintático).

- **Classes minoritárias** (`Organizacao` com 397 exemplos, `Profissao` com
  465) ficam claramente abaixo, com F1 ≈ 0.92 e 0.91 respectivamente. É um
  caso clássico de **desbalanceamento de classes**: o modelo vê estas
  classes ~12× menos vezes que `Pessoa` (5895 exemplos), o que limita a
  sua capacidade de generalização.

- A `Profissao` tem `Recall (0.887)` notavelmente inferior ao `Precision
  (0.937)`. Quando o modelo prevê que algo é profissão, costuma acertar;
  mas falha em identificar muitas profissões reais (falsos negativos). Isto
  é coerente com a hipótese do desbalanceamento — o modelo é "conservador"
  na atribuição desta classe.

- O **spaCy supera o BERT** apesar de ter menos parâmetros, possivelmente
  porque (i) o BERT da aula só foi treinado durante 2 épocas, enquanto o
  spaCy convergiu durante ~20; (ii) o spaCy usa vectores pré-treinados
  específicos para português (`pt_core_news_lg`), bem adaptados ao domínio;
  (iii) o dataset é relativamente pequeno e bem estruturado, condições em
  que arquitecturas mais simples podem competir com transformers.
