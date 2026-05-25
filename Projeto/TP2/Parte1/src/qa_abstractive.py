"""
qa_abstractive.py
-----------------
Módulo de QA abstractivo usando Flan-T5 via prompting.

Ao contrário do QA extrativo (BERT), este módulo gera a resposta
em linguagem natural — não se limita a extrair um span do texto.
Usa o modelo google/flan-t5-base localmente, sem necessidade de API key.
"""

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
MODEL_NAME   = "google/flan-t5-large"  # ~800MB — respostas mais fluentes e abstractivas
MAX_INPUT    = 512                      # tokens máximos de input
MAX_OUTPUT   = 200                      # tokens máximos da resposta gerada
CONTEXT_CHARS = 1800                   # caracteres do contexto a usar no prompt


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class AbstractiveQA:
    """
    QA abstractivo via prompting ao Flan-T5.

    O Flan-T5 foi pré-treinado com instruction tuning, o que significa
    que responde bem a prompts em linguagem natural do tipo:
        "Answer the question based on the context below. ..."
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

    def _build_prompt(self, question: str, context: str) -> str:
        """
        Constrói o prompt para o Flan-T5.
        O Flan-T5 foi treinado com instruction tuning, por isso
        um prompt claro e estruturado produz melhores resultados.
        Trunca o contexto se necessário para caber no limite de tokens.
        """
        # Trunca o contexto a CONTEXT_CHARS caracteres para não exceder MAX_INPUT
        context_trimmed = context[:CONTEXT_CHARS]
        if len(context) > CONTEXT_CHARS:
            context_trimmed += "..."

        prompt = (
            f"Answer the following question based only on the provided context. "
            f"Give a complete and informative answer.\n\n"
            f"Context: {context_trimmed}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        return prompt

    def predict(
        self,
        question: str,
        context: str,
        num_beams: int = 4,
        max_new_tokens: int = MAX_OUTPUT,
    ) -> str:
        """
        Gera uma resposta abstractiva para a pergunta dado o contexto.

        Parameters
        ----------
        question       : str  — pergunta do utilizador
        context        : str  — documento/contexto de suporte
        num_beams      : int  — beam search (maior = melhor qualidade, mais lento)
        max_new_tokens : int  — comprimento máximo da resposta gerada

        Returns
        -------
        str — resposta gerada pelo modelo
        """
        prompt = self._build_prompt(question, context)

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
                max_new_tokens=max_new_tokens,
                num_beams=8,
                early_stopping=True,
                no_repeat_ngram_size=3,   # evita repetições
                length_penalty=1.5,       # favorece respostas mais longas
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()

    def predict_multi_doc(
        self,
        question: str,
        documents: list,
        top_k: int = 3,
    ) -> str:
        """
        Gera resposta combinando contexto de múltiplos documentos.
        Útil quando o retriever retorna vários documentos relevantes.

        Parameters
        ----------
        question  : str        — pergunta do utilizador
        documents : list[dict] — lista de documentos do corpus (com campo 'text')
        top_k     : int        — número de documentos a usar como contexto
        """
        # Concatena os primeiros top_k documentos
        combined_context = "\n\n---\n\n".join(
            f"[{doc['title']}]\n{doc['text'][:600]}"
            for doc in documents[:top_k]
        )
        return self.predict(question, combined_context)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    qa = AbstractiveQA()

    examples = [
        {
            "question": "Where was Cristiano Ronaldo born?",
            "context": (
                "Cristiano Ronaldo dos Santos Aveiro was born on 5 February 1985 "
                "in Funchal, Madeira, Portugal. He is a Portuguese professional "
                "footballer who plays as a forward. Ronaldo has won five UEFA "
                "Champions League titles and has been named the best player in "
                "Europe on multiple occasions."
            ),
        },
        {
            "question": "What makes the Camp Nou stadium special?",
            "context": (
                "Camp Nou is the home stadium of FC Barcelona, located in "
                "Barcelona, Catalonia, Spain. With a seating capacity of over "
                "99,000, it is the largest stadium in Spain and Europe. The "
                "stadium was inaugurated on 24 September 1957 and has hosted "
                "numerous historic matches including the 1982 FIFA World Cup and "
                "the 1999 UEFA Champions League Final. It is widely considered "
                "one of the most iconic football venues in the world."
            ),
        },
        {
            "question": "Why is the offside rule important in football?",
            "context": (
                "Offside is one of the laws in association football, codified in "
                "Law 11 of the Laws of the Game. A player is in an offside "
                "position if any part of the head, body or feet is in the "
                "opponents' half and closer to the opponents' goal line than both "
                "the ball and the second-last opponent. The rule exists to prevent "
                "players from gaining an unfair advantage by positioning themselves "
                "near the opponent's goal and waiting for long passes, which would "
                "reduce the tactical complexity of the game."
            ),
        },
        {
            "question": "How has Messi's career been defined?",
            "context": (
                "Lionel Messi is an Argentine professional footballer widely "
                "regarded as one of the greatest players of all time. He spent "
                "the majority of his career at FC Barcelona, where he won ten "
                "La Liga titles and four UEFA Champions League trophies. Messi "
                "has won the Ballon d'Or eight times, a record. In 2022, he led "
                "Argentina to victory at the FIFA World Cup in Qatar, cementing "
                "his legacy as the best player of his generation."
            ),
        },
    ]

    print("=" * 65)
    print("  DEMO — QA Abstractivo (Flan-T5)")
    print("  Contraste com QA Extrativo: respostas geradas, não extraídas")
    print("=" * 65)

    for ex in examples:
        answer = qa.predict(ex["question"], ex["context"])
        print(f"\n{ex['question']}")
        print(f" {answer}")

    print("\n" + "=" * 65)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo()