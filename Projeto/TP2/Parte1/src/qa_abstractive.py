"""
qa_abstractive.py
-----------------
Modulo de QA abstractivo usando Flan-T5-large via prompting.

Ao contrario do QA extrativo (BERT), este modulo gera a resposta
em linguagem natural -- nao se limita a extrair um span do texto.
Segue o paradigma de prompting a um modelo generativo conforme
descrito no enunciado do trabalho.

Uso:
    python qa_abstractive.py

Requisitos:
    pip install transformers torch
"""

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------
MODEL_NAME  = "google/flan-t5-large"
MAX_INPUT   = 512
MAX_OUTPUT  = 128
INTRO_WORDS = 350   # palavras do inicio do artigo -- intro da Wikipedia


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class AbstractiveQA:
    """
    QA abstractivo via prompting ao Flan-T5-large.

    Usa as primeiras INTRO_WORDS palavras do documento mais relevante
    como contexto. A introducao dos artigos Wikipedia contem sempre
    a informacao factual mais importante (nascimento, fundacao, etc.).
    """

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"A carregar modelo '{model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"   Dispositivo: {self.device}")
        print("Modelo pronto.\n")

    def predict(self, question: str, context: str) -> str:
        """
        Gera uma resposta abstractiva para a pergunta dado o contexto.

        Usa apenas as primeiras INTRO_WORDS palavras do contexto,
        correspondentes a introducao do artigo Wikipedia.

        Parameters
        ----------
        question : str -- pergunta do utilizador
        context  : str -- texto do documento mais relevante

        Returns
        -------
        str -- resposta gerada pelo modelo
        """
        words   = context.split()
        excerpt = " ".join(words[:INTRO_WORDS])

        prompt = (
            f"Based on the context, answer the question with a short factual answer.\n\n"
            f"Context: {excerpt}\n\n"
            f"Question: {question}\n\n"
            f"Short answer:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_INPUT,
            truncation=True,
            padding=False,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_OUTPUT,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    from retriever import Retriever

    print("=" * 65)
    print("  DEMO -- QA Abstractivo (Flan-T5-large)")
    print("=" * 65)

    retriever = Retriever("corpus.json")
    qa        = AbstractiveQA()

    examples = [
        "Where was Cristiano Ronaldo born?",
        "What makes Camp Nou special?",
        "Why is the offside rule important?",
        "How has Messi's career been defined?",
        "When was the UEFA Champions League founded?",
        "What club plays at Anfield?",
        "What year was Ronaldinho born?",
    ]

    for question in examples:
        results  = retriever.search(question, mode="hybrid", top_k=1, use_reranker=True)
        best_doc = results[0][0]
        answer   = qa.predict(question, best_doc["text"])

        print(f"\nQuery   : {question}")
        print(f"Fonte   : {best_doc['title']}")
        print(f"Resposta: {answer}")


if __name__ == "__main__":
    demo()