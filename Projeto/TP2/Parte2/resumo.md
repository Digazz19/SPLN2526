# Attention Is All You Need — Explicação Detalhada

> Vaswani et al., Google Brain / Google Research — NIPS 2017

---

## Índice

1. [Contexto histórico — o que existia antes](#1-contexto-histórico)
2. [A ideia central do artigo](#2-a-ideia-central)
3. [Arquitetura do Transformer](#3-arquitetura-do-transformer)
4. [O mecanismo de atenção em detalhe](#4-o-mecanismo-de-atenção-em-detalhe)
5. [Positional Encoding — como representar a ordem](#5-positional-encoding)
6. [Configuração de treino](#6-configuração-de-treino)
7. [Resultados experimentais](#7-resultados-experimentais)
8. [Generalização a outras tarefas](#8-generalização-a-outras-tarefas)
9. [Porquê a self-attention é melhor](#9-porquê-a-self-attention-é-melhor)
10. [Impacto e legado](#10-impacto-e-legado)

---

## 1. Contexto histórico

Antes do Transformer, o estado da arte em tarefas de **sequence transduction** — o nome técnico para qualquer problema em que transformamos uma sequência de símbolos noutras (tradução, resumo, parsing, etc.) — era dominado por redes neuronais **recorrentes** (RNNs) e, em particular, pelas suas variantes mais sofisticadas: **Long Short-Term Memory (LSTM)** e **Gated Recurrent Units (GRU)**.

### Como funcionavam os RNNs

Imagina que estás a processar a frase *"O gato bebeu o leite"*, token a token. A cada passo de tempo *t*, a rede recebe o token actual *xₜ* e o **estado oculto anterior** *hₜ₋₁*, e produz um novo estado *hₜ*:

```
hₜ = f(hₜ₋₁, xₜ)
```

Este estado *hₜ* é uma espécie de "memória resumida" de tudo o que a rede viu até agora. O problema é que esta memória tem de comprimir toda a informação passada num vector de dimensão fixa, e a informação de tokens muito distantes tende a diluir-se — é o famoso **problema do vanishing gradient** em sequências longas.

### As limitações críticas

**1. Processamento sequencial obrigatório.** Como cada *hₜ* depende de *hₜ₋₁*, é impossível calcular *h₃* sem ter calculado *h₁* e *h₂* antes. Isto significa que, durante o treino, **não se consegue paralelizar** o processamento de uma sequência. Em GPUs modernas, que são desenhadas para fazer muitas operações em simultâneo, este é um desperdício enorme.

**2. Dependências de longa distância.** Em frases como *"O homem que comprou o carro que estava avariado ficou triste"*, o sujeito "homem" e o predicado "ficou triste" estão separados por uma oração relativa longa. Para uma RNN, o sinal que liga "homem" a "ficou" tem de percorrer muitos passos de recorrência, e em cada passo pode degradar-se.

**3. Mecanismos de atenção como remendo.** Já existiam mecanismos de atenção antes deste artigo (por exemplo, no trabalho seminal de Bahdanau et al., 2015, para tradução automática neural), mas eram usados *em cima* de RNNs para mitigar o problema das dependências longas — nunca como substituto completo da recorrência.

---

## 2. A ideia central

A proposta do artigo é radicalmente simples: **eliminar completamente a recorrência e as convoluções**, e construir um modelo baseado *exclusivamente* em mecanismos de atenção.

A intuição é: em vez de tentar guardar toda a informação sobre uma sequência num estado oculto que passa token a token, porque não deixar cada token "olhar directamente" para todos os outros tokens da sequência e decidir quais são relevantes para ele? É exactamente isso que a **self-attention** faz.

Este modelo recebeu o nome de **Transformer**.

---

## 3. Arquitetura do Transformer

O Transformer segue a arquitectura clássica de **encoder-decoder**, usada em tradução automática: o encoder lê a frase de origem, e o decoder gera a frase de destino token a token.

```
Entrada (frase em inglês)
        ↓
   [ ENCODER × 6 ]
        ↓
representações contínuas z
        ↓
   [ DECODER × 6 ]
        ↓
Saída (frase em alemão, gerada auto-regressivamente)
```

### 3.1 O Encoder

O encoder é composto por uma pilha de **N = 6 camadas idênticas**. Cada camada tem exactamente duas sub-camadas:

**Sub-camada 1 — Multi-Head Self-Attention:**
Cada posição da sequência de entrada pode atender a todas as outras posições da mesma sequência. Isto permite capturar relações entre quaisquer dois tokens, independentemente da sua distância.

**Sub-camada 2 — Feed-Forward Network (posição a posição):**
Uma rede neuronal feed-forward simples aplicada **independentemente** a cada posição:

```
FFN(x) = max(0, xW₁ + b₁) W₂ + b₂
```

A dimensão interna é *d_ff = 2048*, enquanto a dimensão de entrada e saída é *d_model = 512*.

**Residual connections e Layer Normalization:**
Depois de cada sub-camada, é adicionada uma **residual connection** (a ideia de He et al., 2016, em redes de visão) e aplica-se **layer normalization**:

```
saída = LayerNorm(x + Sublayer(x))
```

A residual connection ajuda os gradientes a fluir durante o backpropagation em redes profundas. A layer normalization estabiliza o treino ao normalizar as activações dentro de cada camada.

### 3.2 O Decoder

O decoder também tem **N = 6 camadas**, mas cada camada tem **três** sub-camadas:

**Sub-camada 1 — Masked Multi-Head Self-Attention:**
Idêntica à self-attention do encoder, mas com uma diferença crucial: o **masking**. Durante o treino, o decoder precisa de garantir que a previsão para a posição *i* só depende das posições anteriores (0 a *i-1*), nunca das futuras — caso contrário o modelo "faria batota" olhando para a resposta. Isto é implementado colocando *-∞* nas entradas do softmax correspondentes a posições futuras (depois do softmax, *e^(-∞) = 0*, logo esses tokens têm peso zero).

**Sub-camada 2 — Encoder-Decoder Attention (ou Cross-Attention):**
Aqui as *queries* vêm da sub-camada anterior do decoder, mas as *keys* e *values* vêm da saída do encoder. Isto é o mecanismo que permite ao decoder "consultar" a frase de origem enquanto gera a frase de destino — é a ponte entre as duas metades do modelo.

**Sub-camada 3 — Feed-Forward Network:**
Igual à do encoder.

### 3.3 Embeddings e geração de probabilidades

Tanto as palavras de entrada como as de saída são convertidas em vectores de dimensão *d_model = 512* por camadas de embedding aprendidas. O decoder termina com uma camada linear e um softmax que converte o vector de dimensão *d_model* numa distribuição de probabilidade sobre todo o vocabulário.

Uma técnica interessante do artigo: as **matrizes de embedding de entrada e saída partilham os mesmos pesos** (e os mesmos pesos da camada linear final). Isto reduz o número de parâmetros e foi mostrado por Press & Wolf (2016) ser benéfico.

---

## 4. O mecanismo de atenção em detalhe

Esta é a parte mais importante do artigo. Convém perceber bem o que se passa matematicamente.

### 4.1 O que é a atenção?

A ideia abstracta é simples. Tens uma **query** (o que estás a procurar), um conjunto de **keys** (etiquetas para o que está disponível), e um conjunto de **values** (o conteúdo actual). Queres recuperar uma combinação dos valores, pesada por quão bem a tua query corresponde a cada key.

Uma analogia: imagina uma base de dados onde cada entrada tem uma chave (key) e um valor (value). A tua query não tem de corresponder exactamente a nenhuma key — em vez disso, calculas uma pontuação de compatibilidade entre a query e cada key, e retornas uma média ponderada de todos os values.

### 4.2 Scaled Dot-Product Attention

O mecanismo concreto usado no artigo:

```
Attention(Q, K, V) = softmax( Q Kᵀ / √dₖ ) V
```

Onde:
- **Q** (queries) é uma matriz de dimensão *(nq × dₖ)*
- **K** (keys) é uma matriz de dimensão *(nk × dₖ)*
- **V** (values) é uma matriz de dimensão *(nk × dᵥ)*

**Passo a passo:**

1. **Calcula os dot products** entre cada query e todas as keys: *Q Kᵀ* produz uma matriz de pontuações de dimensão *(nq × nk)*. Cada entrada *(i, j)* diz quão compatível é a query *i* com a key *j*.

2. **Escala por √dₖ**: divide cada pontuação por *√dₖ*. Porquê? Se *dₖ* for grande, os dot products tendem a crescer em magnitude (variância de *dₖ* se os componentes tiverem variância 1), empurrando o softmax para regiões de gradiente muito pequeno. A escala resolve isso.

3. **Aplica softmax**: converte as pontuações em pesos que somam 1. Isto diz, para cada query, quanto peso dar a cada value.

4. **Multiplica pelos values**: a saída é uma média ponderada dos values, onde os pesos são as atenções calculadas.

### 4.3 Porquê não usar additive attention?

Existe outra formulação popular, a **additive attention** (Bahdanau et al., 2015), que usa uma rede feed-forward de uma camada para calcular a compatibilidade:

```
score(q, k) = vᵀ tanh(Wq q + Wk k)
```

A diferença prática é que o **dot-product attention é muito mais rápido** e eficiente em memória porque pode ser implementado com multiplicação matricial altamente optimizada. Teoricamente têm complexidade similar, mas na prática a diferença é enorme com GPUs modernas.

### 4.4 Multi-Head Attention — a grande inovação

Em vez de aplicar um único mecanismo de atenção com dimensão *d_model*, o artigo propõe projectar as queries, keys e values **h vezes** com diferentes projecções lineares aprendidas, aplicar atenção em paralelo em cada "cabeça" (head), e concatenar os resultados:

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) Wᴼ

onde headᵢ = Attention(Q Wᵢᴼ, K Wᵢᴷ, V Wᵢᵛ)
```

As matrizes de projecção são:
- *Wᵢᴼ ∈ ℝ^(d_model × dₖ)* — projecção das queries
- *Wᵢᴷ ∈ ℝ^(d_model × dₖ)* — projecção das keys
- *Wᵢᵛ ∈ ℝ^(d_model × dᵥ)* — projecção dos values
- *Wᴼ ∈ ℝ^(h·dᵥ × d_model)* — projecção final

Com **h = 8** cabeças e *d_model = 512*, cada cabeça tem *dₖ = dᵥ = 512/8 = 64*.

**Porquê múltiplas cabeças?** Com uma única cabeça de atenção, o output é uma média das representações de todos os tokens ponderada pelas atenções — e esta média "dilui" a informação. Com múltiplas cabeças, cada uma pode focar em aspectos diferentes da sequência: uma cabeça pode aprender a capturar relações sintácticas, outra semânticas, outra correferências, etc. Os autores mostram nos apêndices que as cabeças de facto aprendem comportamentos distintos de forma espontânea.

### 4.5 Os três usos da atenção no Transformer

O modelo usa atenção de três formas distintas:

**1. Self-attention no encoder:** Queries, keys e values vêm todos da camada anterior do encoder. Cada posição pode atender a todas as outras posições da frase de entrada.

**2. Masked self-attention no decoder:** Igual ao anterior, mas com masking para impedir atenção a posições futuras. Necessário para manter a propriedade auto-regressiva.

**3. Cross-attention (encoder-decoder):** As queries vêm do decoder, mas as keys e values vêm do encoder. Permite ao decoder consultar toda a sequência de entrada enquanto gera a saída.

---

## 5. Positional Encoding

Aqui há um problema subtil mas crítico: ao contrário de uma RNN, a self-attention é **invariante à permutação**. Ou seja, se trocares a ordem dos tokens na entrada, o resultado da atenção seria o mesmo (porque cada token pode atender a todos os outros com igual facilidade). O modelo não sabe intrinsecamente que "gato" vem antes de "bebeu".

Para resolver isto, o artigo injjecta **informação de posição** nas embeddings antes de entrarem no encoder e decoder. Especificamente, adiciona-se um vector de posição *PE* de dimensão *d_model* a cada embedding:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Onde *pos* é a posição na sequência e *i* é o índice da dimensão.

**Porquê senos e cossenos?** Há várias razões:

1. **Extrapolação para comprimentos maiores:** Uma rede treinada com sequências de até 512 tokens consegue, em princípio, generalizar para sequências mais longas, porque as funções sinusoidais têm padrões regulares e previsíveis.

2. **Relações relativas aprendíveis:** Para qualquer offset fixo *k*, *PE(pos+k)* pode ser expresso como uma transformação linear de *PE(pos)*. Isto significa que o modelo pode aprender facilmente a atender a "o token que está 3 posições antes de mim", por exemplo.

3. **Cada posição tem uma representação única:** Como as frequências são diferentes para cada dimensão, o vector de posição é único para cada *pos*.

Os autores também experimentaram com **positional embeddings aprendidos** (em vez de fixos) e obtiveram resultados quase idênticos — o que sugere que a forma específica do encoding não é crítica, desde que forneça informação de posição.

---

## 6. Configuração de treino

### 6.1 Dados

- **EN→DE:** WMT 2014 English-German, ~4.5 milhões de pares de frases. Vocabulário partilhado de ~37.000 tokens usando **BPE** (Byte-Pair Encoding).
- **EN→FR:** WMT 2014 English-French, ~36 milhões de frases. Vocabulário de 32.000 word-pieces.

O BPE é relevante para a cadeira: é o algoritmo de tokenização sub-palavra que vimos nas aulas, que divide palavras raras em subunidades mais frequentes (por exemplo, *"unhappiness"* → *"un"*, *"happiness"*), equilibrando vocabulário e cobertura.

### 6.2 Hardware

Uma única máquina com **8 GPUs NVIDIA P100**. O modelo base treinou em ~12 horas, o modelo big em ~3.5 dias.

### 6.3 Optimizer

Adam com *β₁=0.9*, *β₂=0.98*, *ε=10⁻⁹*, e um **learning rate schedule** personalizado:

```
lrate = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))
```

Com *warmup_steps = 4000*. Isto significa:
- Nas primeiras 4000 steps, o learning rate **aumenta linearmente** de 0.
- A partir daí, **decresce** proporcionalmente a *1/√step*.

Esta estratégia de warmup é importante: nas primeiras iterações, os gradientes são muito grandes e ruidosos; um learning rate pequeno no início evita actualizações destabilizadoras.

### 6.4 Regularização

**Residual Dropout (*P_drop = 0.1*):** Aplicado ao output de cada sub-camada antes da adição ao residual, e também às somas de embeddings + positional encodings. Impede overfitting ao "desligar" aleatoriamente 10% das activações durante o treino.

**Label Smoothing (*ε_ls = 0.1*):** Em vez de treinar com targets *one-hot* (100% de probabilidade na palavra correcta), suaviza a distribuição: 90% na palavra correcta, 10% distribuído uniformemente pelas restantes. Isto *aumenta* a perplexidade (o modelo é forçado a ser menos confiante), mas *melhora* o BLEU score real — o modelo aprende a ser mais calibrado e a explorar alternativas.

### 6.5 Inferência

**Beam search** com *beam size = 4* e penalização de comprimento *α = 0.6*. O comprimento máximo de saída é comprimento de entrada + 50 (com terminação antecipada possível).

Para o modelo base, calcula-se a média dos **últimos 5 checkpoints** (guardados de 10 em 10 minutos). Para o modelo big, dos **últimos 20 checkpoints**. Esta técnica de checkpoint averaging é barata e melhora consistentemente a qualidade.

---

## 7. Resultados experimentais

### 7.1 Tradução automática EN→DE

| Modelo | BLEU | FLOPs |
|---|---|---|
| ConvS2S Ensemble | 26.36 | 7.7 × 10¹⁹ |
| GNMT+RL Ensemble | 26.30 | 1.8 × 10²⁰ |
| **Transformer (base)** | **27.3** | **3.3 × 10¹⁸** |
| **Transformer (big)** | **28.4** | **2.3 × 10¹⁹** |

O Transformer (big) supera o **melhor ensemble** anterior por mais de 2 pontos BLEU — um salto enorme nesta métrica. E o modelo base já bate todos os sistemas anteriores com **apenas ~4% do custo computacional** do ConvS2S Ensemble.

### 7.2 Tradução automática EN→FR

| Modelo | BLEU | FLOPs |
|---|---|---|
| Deep-Att + PosUnk Ensemble | 40.4 | 8.0 × 10²⁰ |
| GNMT+RL Ensemble | 41.16 | 1.1 × 10²¹ |
| ConvS2S Ensemble | 41.29 | 1.2 × 10²¹ |
| **Transformer (big)** | **41.8** | **2.3 × 10¹⁹** |

Novo SOTA como **modelo único** (não ensemble), com menos de 1/4 do custo do modelo anterior mais eficiente.

### 7.3 Ablation study — o que importa?

Os autores fizeram variações sistemáticas do modelo base para perceber o impacto de cada componente. Os resultados mais interessantes:

**Número de cabeças (h):**
- h=1 (single-head): BLEU 24.9 — pior 0.9 pontos que h=8
- h=8: BLEU 25.8 — óptimo
- h=16: BLEU 25.8 — sem melhoria com mais cabeças
- h=32: BLEU 25.4 — pior, provavelmente porque *dₖ = 16* é demasiado pequeno

**Dimensão dₖ:**
- Reduzir *dₖ* de 64 para 16 degrada de 25.8 para 25.1 BLEU
- Isto sugere que determinar compatibilidade entre queries e keys é difícil, e precisa de dimensão suficiente

**Tamanho do modelo:**
- Modelos maiores são consistentemente melhores (mais parâmetros → mais capacidade)

**Dropout:**
- Sem dropout: BLEU desce de 25.8 para 24.6
- Dropout é crítico — o modelo é suficientemente grande para overfitting sem regularização

**Positional encoding:**
- Sinusoidal vs aprendido: ambos dão BLEU ~25.7-25.8 — essencialmente idênticos

---

## 8. Generalização a outras tarefas

Para mostrar que o Transformer não é um modelo específico para tradução, os autores aplicam-no a **English constituency parsing** (análise sintáctica de constituintes) no Penn Treebank (secção WSJ).

Esta tarefa é desafiante por razões diferentes da tradução:
- O output tem **constraints estruturais fortes** (tem de ser uma árvore sintáctica válida)
- O output é **significativamente mais longo** que o input
- Os modelos RNN sequence-to-sequence tinham dificuldade nesta tarefa com poucos dados

Configuração: Transformer de 4 camadas com *d_model = 1024*, **sem qualquer ajuste específico** para parsing.

| Modelo | Treino | F1 (WSJ Section 23) |
|---|---|---|
| Dyer et al. (2016) — generativo | WSJ | 93.3 |
| **Transformer 4L** | **semi-supervisionado** | **92.7** |
| Vinyals et al. (2014) | semi-supervisionado | 92.1 |
| Transformer 4L | WSJ only | 91.3 |
| BerkeleyParser (Petrov et al.) | WSJ | 90.4 |

O Transformer supera o BerkeleyParser e quase todos os modelos anteriores, apesar de não ter sido ajustado para esta tarefa. Isto demonstra que a arquitectura aprende representações generalizáveis.

---

## 9. Porquê a self-attention é melhor

Os autores fazem uma análise teórica comparando self-attention, recorrência e convoluções em três dimensões:

### 9.1 Complexidade computacional por camada

| Tipo | Complexidade |
|---|---|
| Self-Attention | O(n² · d) |
| Recurrent | O(n · d²) |
| Convolutional (kernel k) | O(k · n · d²) |

Para sequências típicas em tradução (*n* < *d*), a self-attention é mais eficiente que a recorrência. Para sequências muito longas, poderia ser problemático, mas os autores propõem uma variante restrita com vizinhança de tamanho *r* que reduz para O(r·n·d).

### 9.2 Operações sequenciais mínimas

| Tipo | Ops. Sequenciais |
|---|---|
| Self-Attention | O(1) |
| Recurrent | O(n) |
| Convolutional | O(1) |

Este é o factor mais importante para paralelização. A recorrência exige *n* passos sequenciais, limitando fundamentalmente a eficiência em GPUs. A self-attention e as convoluções podem ser calculadas em paralelo.

### 9.3 Path length entre dependências de longa distância

Esta é a dimensão mais subtil. O **path length** mede quantos passos (de atenção, recorrência ou convolução) um sinal tem de percorrer para ir de uma posição a outra.

| Tipo | Path Length |
|---|---|
| Self-Attention | O(1) |
| Recurrent | O(n) |
| Convolutional (kernel k) | O(n/k) ou O(log_k n) |

Com self-attention, qualquer dois tokens comunicam directamente em 1 passo — independentemente de estarem adjacentes ou nos extremos opostos de uma frase longa. Isto facilita enormemente a aprendizagem de dependências de longa distância.

### 9.4 Interpretabilidade como subproduto

Um benefício inesperado da self-attention é que produz modelos mais interpretáveis. Os autores mostram visualizações das distribuições de atenção que revelam comportamentos linguisticamente coerentes:

- Algumas cabeças aprendem a fazer **resolução de anáfora**: quando o modelo processa "its", várias cabeças atendem fortemente a "Law" (o antecedente de "its application").
- Outras cabeças capturam **dependências sintácticas de longa distância**: para a palavra "making", as atenções vão para "more difficult", completando o sintagma "making...more difficult" mesmo que estejam separados por vários tokens.
- Cabeças diferentes claramente aprenderam a executar **tarefas diferentes** sem supervisão explícita.

---

## 10. Impacto e legado

Este artigo é provavelmente o mais influente em Processamento de Linguagem Natural da última década. As suas consequências directas foram:

**BERT (2018):** Usa apenas o encoder do Transformer, treinado com masked language modeling. Tornou-se o modelo de referência para NLP durante vários anos.

**GPT (2018) e família GPT-2, GPT-3, GPT-4:** Usa apenas o decoder do Transformer (com masked self-attention). A base de praticamente todos os modelos de linguagem generativos actuais.

**T5 (2019):** Usa o Transformer encoder-decoder completo, formulando todas as tarefas NLP como sequence-to-sequence.

**Vision Transformers (ViT, 2020):** Aplica o Transformer a imagens, tratando patches como tokens — exactamente o que os autores anteciparam em trabalho futuro.

**Diffusion Transformers (DiT, 2022):** Base de modelos de geração de imagem como Stable Diffusion 3 e DALL-E 3.

---

## Ligação ao SPLN

Para contextualizar no que foi estudado na cadeira:

**BPE (Byte-Pair Encoding):** É o algoritmo de tokenização usado nos experimentos deste artigo (vocabulário de 37K tokens para EN-DE). O BPE resolve o problema de palavras fora do vocabulário e é a base dos tokenizadores modernos (BPE, WordPiece, SentencePiece).

**N-gram Language Models:** Os N-gram LMs que estudámos são o baseline clássico que o Transformer supera a nível de perplexidade. A perplexidade do modelo base neste artigo é 4.92 por word-piece, muito inferior ao que se obtém com N-gramas.

**Word Embeddings / Word2Vec:** As camadas de embedding do Transformer são directamente análogas ao que o Word2Vec aprende — representações vectoriais densas de tokens. A diferença é que no Transformer as embeddings são *contextuais*: a representação de "banco" muda consoante o contexto.

**NER e classificação com BERT:** Os modelos de NER fine-tuned (como fizemos no TPC5 com `neuralmind/bert-base-portuguese-cased`) são directamente baseados neste artigo — o BERT é um encoder Transformer pré-treinado.

**TF-IDF e recuperação de informação:** Os sistemas de IR tradicionais como o TF-IDF foram largamente substituídos por encoders Transformer que geram embeddings densas para pesquisa semântica (dense retrieval).

---

*Referência:* Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.