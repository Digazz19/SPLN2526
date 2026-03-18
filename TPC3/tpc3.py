import spacy
from collections import defaultdict, Counter
import itertools

nlp = spacy.load("pt_core_news_sm")

f = open("Harry_Potter_e_a_Pedra_Filosofal.txt", "r", encoding="utf-8")

text = f.read()

doc = nlp(text)

contagem = Counter()
for ent in doc.ents:
    if ent.label_ == "PER" and ent.text[0].isupper():
        nome = ent.text.split()[0]
        contagem[nome] += 1

cooccurrence = defaultdict(int)
for sent in doc.sents:
    personagens = set()
    for ent in sent.ents:
        if ent.label_ == "PER" and ent.text[0].isupper():
            nome = ent.text.split()[0]
            personagens.add(nome)

    for a, b in itertools.combinations(personagens, 2):
        pair = tuple(sorted([a, b]))
        cooccurrence[pair] += 1

with open("output.txt", "w", encoding="utf-8") as f:
    f.write("=== PERSONAGENS ===\n")
    for nome, n in contagem.most_common():
        f.write(f"{nome}: {n}\n")
    
    f.write("\n=== CO-OCORRÊNCIAS ===\n")
    for (a, b), w in sorted(cooccurrence.items(), key=lambda x: -x[1]):
        f.write(f"{a} -- {b}: {w}\n")

