from groq import Groq

class SemanticSummarizer:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.model = 'llama-3.1-8b-instant' 
        self.chunk_size = 1800 

    def summarize(self, text):
        if not text:
            return "No text to summarize."

        chunks = self._split_text(text)
        
        if len(chunks) == 1:
            return self._summarize_chunk(chunks[0])

        detailed_summaries = []
        
        for chunk in chunks:
            summary_part = self._summarize_chunk(chunk)
            
            # If chunk returns empty or "No Data", skip it to keep summary clean
            if summary_part and len(summary_part) > 20:
                detailed_summaries.append(summary_part)

        if not detailed_summaries:
            return "Could not generate summary."

        # Join with delimiter
        return "---TOPIC---".join(detailed_summaries)

    def _summarize_chunk(self, text):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a STRICT Fact Extractor. "
                            "Extract ONLY concrete information from the provided text."
                            "\nSTRICT NEGATIVE RULES (Must Follow):"
                            "\n1. FORBIDDEN TOPICS: Do NOT write about 'Climate Change', 'Global Warming', 'Food Security', 'Greenhouse Gases' or 'Sustainability'."
                            "\n2. FORBIDDEN BEHAVIOR: Do NOT vent, ramble, or invent scenarios about these topics. If the text provided does NOT explicitly mention these topics, do NOT mention them."
                            "\n3. INTRODUCTIONS: If the text is just an introduction (e.g., 'This report outlines...'), return 'Introduction (No Data)'."
                            "\n4. FORMAT: Return ## [Section Name] and bullet points for facts."
                            "\n5. FACT-CHECK: Use ONLY facts explicitly stated in the text. Do not use outside knowledge."
                            "\n6. START: Begin immediately with the first '##' heading."
                        )
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                model=self.model,
                temperature=0.0, # Temperature 0 ensures AI is factual, not creative
            )
            raw_output = chat_completion.choices[0].message.content
            
            # Clean output: Strip anything before first Header
            if "##" in raw_output:
                return raw_output[raw_output.index("##"):]
            return raw_output

        except Exception as e:
            return f"Error: {str(e)}"

    def _split_text(self, text):
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += "\n\n" + para
            else:
                chunks.append(current_chunk.strip())
                current_chunk = para
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks