# Harry Potter e a Pedra Filosofal — Rede de Personagens

Extrai personagens de um livro em português usando **spaCy** e calcula as suas relações com base em **co-ocorrência por frase**.

## Como funciona

1. O texto é processado pelo modelo NER `pt_core_news_sm`
2. As entidades do tipo `PER` são extraídas e contadas
3. Para cada frase, todos os pares de personagens que aparecem juntos são registados
4. O resultado é guardado num ficheiro `output.txt`

## Output

O ficheiro `output.txt` gerado tem duas secções:

```
=== PERSONAGENS ===
Rony: 394
Hagrid: 298
Hermione: 267
...

=== CO-OCORRÊNCIAS ===
Hermione -- Rony: 72
Fred -- Jorge: 15
...
```

- **PERSONAGENS** — lista de todos os nomes reconhecidos como `PER`, ordenados por número de menções
- **CO-OCORRÊNCIAS** — pares de personagens que aparecem na mesma frase, ordenados por frequência