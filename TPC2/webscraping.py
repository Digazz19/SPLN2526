import requests
from bs4 import BeautifulSoup
import json
import string
import time

base_url = "https://www.atlasdasaude.pt"

res = {}

for letra in string.ascii_lowercase:
    link = f"https://www.atlasdasaude.pt/doencasAaZ/{letra}"

    html_doc = requests.get(link)
    soup = BeautifulSoup(html_doc.text, "html.parser")

    doencas_div = soup.find_all("div", class_="views-row")

    for div in doencas_div:
        designacao = div.div.h3.a.text.strip()

        link_rel = div.find("h3").a["href"]
        fulllink = base_url + link_rel

        desc_tag = div.find("div", class_="views-field-body")
        descricaopeq = desc_tag.text.strip() if desc_tag else ""

        page = requests.get(fulllink)
        soup2 = BeautifulSoup(page.text, "html.parser")

        body = soup2.find("div", class_="field-name-body")

        descricao = ""
        causas = ""
        sintomas = []
        tratamento = ""

        if body:
            section = "descricao"

            for tag in body.find_all(["h2", "p", "ul"]):

                if tag.name == "h2":
                    titulo = tag.text.lower()

                    if "causa" in titulo:
                        section = "causas"
                    elif "sintoma" in titulo:
                        section = "sintomas"
                    elif "tratamento" in titulo:
                        section = "tratamento"

                elif tag.name == "p":
                    if section == "descricao":
                        descricao += tag.text.strip() + " "
                    elif section == "causas":
                        causas += tag.text.strip() + " "
                    elif section == "tratamento":
                        tratamento += tag.text.strip() + " "

                elif tag.name == "ul" and section == "sintomas":
                    for li in tag.find_all("li"):
                        sintomas.append(li.text.strip())

        res[designacao] = {
            "descricao_pequena": descricaopeq,
            "descricao": descricao.strip(),
            "causas": causas.strip(),
            "sintomas": sintomas,
            "tratamento": tratamento.strip(),
            "link": fulllink
        }

        time.sleep(0.5)  # para não sobrecarregar o site

# guardar json
with open("doencas.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=4)

print("Dataset criado com", len(res), "doenças")