import numpy as np

class SemanticQA:
    def __init__(self, model):
        self.model = model
        self.context_chunks = []
        self.chunk_embeddings = None

    def prepare_context(self, text):
        """
        Splits text into chunks and computes embeddings for retrieval.
        """
        # Create chunks (paragraphs or roughly 3 sentences)
        raw_chunks = text.split('\n\n')
        self.context_chunks = [c.strip() for c in raw_chunks if len(c.strip()) > 50]
        
        # If the PDF didn't have paragraphs, fall back to fixed length splitting
        if not self.context_chunks:
            sentences = text.split('. ')
            chunk = []
            for sent in sentences:
                chunk.append(sent)
                if len(' '.join(chunk)) > 300:
                    self.context_chunks.append('. '.join(chunk))
                    chunk = []
            if chunk:
                self.context_chunks.append('. '.join(chunk))

        if not self.context_chunks:
            self.context_chunks = [text]

        # Embed chunks
        self.chunk_embeddings = self.model.encode(self.context_chunks, convert_to_tensor=False)

    def ask(self, question):
        """
        Finds the most relevant chunk to answer the question.
        """
        if not self.context_chunks:
            return "I'm sorry, I couldn't process the document content."

        # Embed the question
        query_embedding = self.model.encode([question], convert_to_tensor=False)

        # Calculate cosine similarity (dot product for normalized vectors)
        scores = np.dot(self.chunk_embeddings, query_embedding.T).flatten()

        # Get index of highest score
        best_idx = np.argmax(scores)
        
        return self.context_chunks[best_idx]