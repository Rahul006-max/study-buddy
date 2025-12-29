import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

class SemanticQA:
    def __init__(self, model, api_key):
        self.model = model
        self.client = Groq(api_key=api_key)
        self.chunk_embeddings = None
        self.context_chunks = []

    def prepare_context(self, text):
        self.context_chunks = [c.strip() for c in text.split('\n\n') if len(c.strip()) > 50]
        if not self.context_chunks:
            self.context_chunks = [text]
        self.chunk_embeddings = self.model.encode(self.context_chunks, convert_to_tensor=False)

    def ask(self, question):
        if not self.context_chunks:
            return "I'm sorry, I couldn't process the document content."

        # 1. Embed Question
        query_embedding = self.model.encode([question], convert_to_tensor=False)

        # 2. Similarity Search (Dot Product)
        scores = np.dot(self.chunk_embeddings, query_embedding.T).flatten()

        # 3. RETRIEVE TOP 3 MATCHES (Multi-Context Strategy)
        # Instead of taking the single best, we take top 3 to give AI more info.
        top_k_indices = np.argsort(scores)[::-1][:3]
        best_contexts = [self.context_chunks[i] for i in top_k_indices]

        # 4. Combine Contexts
        # We present all 3 pieces to the AI so it can synthesize the best answer
        context_text = "\n\n".join([f"[Source {i+1}]: {ctx}" for i, ctx in enumerate(best_contexts)])

        # 5. Generate Answer with System Instruction
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise document analyzer. "
                            "Answer the user's question based ONLY on the provided context segments."
                            "\nINSTRUCTIONS:"
                            "\n1. SYNTHESIZE: If information in Source 1 contradicts Source 2, reconcile it."
                            "\n2. ACCURACY: If the answer is not in the context, say 'The text does not contain this information'."
                            "\n3. COMPLETE: Provide a comprehensive answer using all sources."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Question: {question}\n\nContext Segments:\n{context_text}"
                    }
                ],
                model='llama-3.1-8b-instant',
                temperature=0.1, # Low temp for factual accuracy
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error retrieving answer: {str(e)}"