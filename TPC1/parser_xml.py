import re

with open("medicina.xml", "r", encoding="utf-8") as f:
    xml = f.read()

# extrair conteúdo das tags <text>
texts = re.findall(r"<text[^>]*>(.*?)</text>", xml, re.DOTALL)

# remover tags internas (<i>, <b>, etc.)
texts = [re.sub(r"<.*?>", "", t) for t in texts]

# juntar tudo num único texto
text = "\n".join(t.strip() for t in texts if t.strip())

text = re.sub(r"\n(\d+) ", r"@\1 ", text)

concepts = re.split(r"@", text)

def process_concepts(c):

    c = re.sub(r"SIN\.-", r"@SIN.-", c)
    c = re.sub(r"VAR\.-", r"@VAR.-", c)
    c = re.sub(r"Nota\.-", r"@NOTA.-", c)

    c = re.sub(r"(?m)^(en|pt|es|la)\s", r"#\1 ", c)

    id = re.search(r"^(\d+)", c)
    term = re.search(r"^\d+\s+([^\n@]+)", c)
    sin = re.search(r"@SIN\.-([^@#]+)", c)
    var = re.search(r"@VAR\.-([^@#]+)", c)
    nota = re.search(r"@NOTA\.-([^@#]+)", c)
    es = re.search(r"#es\s([^@#]+)", c)
    en = re.search(r"#en\s([^@#]+)", c)
    pt = re.search(r"#pt\s([^@#]+)", c)
    la = re.search(r"#la\s([^@#]+)", c)

    def extract_translations(match):
        if not match:
            return []
        
        text = match.group(1)

        # remover quebras de linha
        text = re.sub(r"\s*\n\s*", " ", text)

        return [t.strip() for t in text.split(";")]

    es_list = extract_translations(es)
    en_list = extract_translations(en)
    pt_list = extract_translations(pt)
    la_list = extract_translations(la)

    print("ID:", id.group(1) if id else "N/A")
    print("TERM:", term.group(1).strip() if term else "N/A")
    print("SIN:", sin.group(1).strip() if sin else "N/A")
    print("VAR:", var.group(1).strip()  if var else "N/A")
    print("NOTA:", nota.group(1).strip()  if nota else "N/A")
    
    print("ES:", es_list)
    print("EN:", en_list)
    print("PT:", pt_list)
    print("LA:", la_list)

for c in concepts:
    print("Concept ------------------")
    process_concepts(c)
