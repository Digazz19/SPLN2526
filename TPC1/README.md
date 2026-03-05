# Parser de Vocabulário Médico (XML)

## Descrição

Este trabalho consiste na implementação de um **parser em Python** para extrair informação de um ficheiro XML gerado a partir de um PDF de vocabulário médico.

O objetivo é utilizar **expressões regulares (biblioteca `re`)** para identificar e extrair diferentes elementos presentes no ficheiro, organizando a informação de cada conceito.

O ficheiro XML contém múltiplas entradas com termos médicos e respetivas traduções em várias línguas.

---

## Estrutura do XML

O ficheiro XML contém elementos `<text>` que representam fragmentos de texto extraídos do PDF.

Exemplo:

```xml
<text top="401" left="149" width="44" height="14" font="7"><i>abdomen</i></text>
```

Estes elementos são extraídos e combinados para reconstruir o conteúdo textual original.

---

## Funcionalidades do Parser

O script realiza os seguintes passos:

### 1. Leitura do ficheiro XML

O ficheiro `medicina.xml` é carregado para memória.

### 2. Extração do texto

Utiliza-se uma expressão regular para extrair o conteúdo das tags `<text>`.

```python
re.findall(r"<text[^>]*>(.*?)</text>", xml, re.DOTALL)
```

### 3. Limpeza de tags internas

São removidas tags internas como:

- `<i>`
- `<b>`

### 4. Reconstrução do texto

Os fragmentos de texto são concatenados para reconstruir o conteúdo original.

### 5. Separação de conceitos

Cada conceito é identificado pelo seu **ID numérico** e separado usando expressões regulares.

### 6. Extração da informação

Para cada conceito são extraídos os seguintes campos:

- **ID**
- **Termo principal**
- **SIN** (sinónimos)
- **VAR** (variantes)
- **NOTA**
- **Traduções nas diferentes línguas**
  - Espanhol (`es`)
  - Inglês (`en`)
  - Português (`pt`)
  - Latim (`la`)

### 7. Tratamento das traduções

Algumas traduções aparecem separadas por `;`.  
Estas são divididas em listas.

Exemplo:

```
abdomen; abdominal region; gut
```

↓

```
['abdomen', 'abdominal region', 'gut']
```

### 8. Limpeza de quebras de linha

O XML pode conter quebras de linha (`\n`) dentro dos conceitos.  
Estas são removidas para evitar quebras incorretas nas traduções.

---

## Exemplo de Output

```
Concept ------------------
ID: 5346
TERM: xemelgos siameses
SIN: siameses; xemelgos unidos
VAR: N/A
NOTA: N/A

ES: ['gemelos siameses', 'gemelos unidos', 'siameses']
EN: ['conjoined twins', 'siamese twins']
PT: ['gêmeos siameses [Br.]', 'gémeos siameses [Pt.]']
LA: []
```

---

## Como executar

1. Colocar o ficheiro XML na mesma pasta do script.

2. Executar o programa:

```bash
python parser_xml.py
```

O script irá imprimir no terminal os conceitos extraídos.
