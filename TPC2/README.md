# Atlas da Saúde - Web Scraper de Doenças

## Descrição

Este trabalho consiste num **web scraper em Python** que recolhe informação sobre várias doenças disponíveis no site **Atlas da Saúde**.

O script percorre todas as páginas de doenças organizadas alfabeticamente e extrai informação relevante de cada uma delas, criando automaticamente um **dataset em formato JSON**.

---

## Dados extraídos

Para cada doença são recolhidas as seguintes informações:

* **designação** – nome da doença
* **descrição pequena** – pequeno resumo presente na lista de doenças
* **descrição** – descrição detalhada da doença
* **causas** – possíveis causas da doença
* **sintomas** – lista de sintomas associados
* **tratamento** – possíveis tratamentos
* **link** – link da página original da doença

Todos os dados são guardados no ficheiro [doencas.json](doencas.json)

---

## Funcionamento do Script

O script segue os seguintes passos:

1. Percorre todas as letras do alfabeto (`a-z`).
2. Para cada letra, acede à página correspondente de doenças:

```
https://www.atlasdasaude.pt/doencasAaZ/<letra>
```

3. Extrai todas as doenças listadas nessa página.

4. Para cada doença:

   * obtém o **nome**
   * obtém a **descrição pequena**
   * recolhe o **link para a página completa**

5. Acede à página individual de cada doença e extrai:

   * descrição
   * causas
   * sintomas
   * tratamento

6. Guarda os dados num **dicionário Python**.

7. No final, exporta toda a informação para um ficheiro **JSON**.

---

## Instalar dependências

```bash
pip install requests beautifulsoup4
```

## Executar o script

```bash
python webscraping.py
```

Após a execução será criado o ficheiro [doencas.json](doencas.json)

---

## ⚠️ Nota

O script inclui um pequeno atraso entre pedidos:

```python
time.sleep(0.5)
```

Isto evita sobrecarregar o servidor do website durante o scraping.

---
