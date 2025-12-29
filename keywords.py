from groq import Groq

class ConceptExtractor:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.model = 'llama-3.1-8b-instant'

    def extract_keywords(self, text):
        if not text:
            return []
        
        # Use a larger sample of summary (up to 3000 chars) for better context
        context = text[:3000]

        prompt = (
            f"Extract the 5-8 most central topics, technologies, or entities from the text below. "
            f"Return them as a comma-separated list."
            f"\nSTRICT NEGATIVE RULES (Must Follow):"
            f"\n1. IGNORE HEADERS: Do NOT include generic section headers like 'Introduction', 'Overview', 'Module 1', 'Chapter 1' unless they contain actual specific content."
            f"\n2. FOCUS: Identify specific, concrete subjects (e.g., 'Power BI', 'Data Visualization', 'Spatial Data')."
            f"\n3. FORMATTING: Do NOT number them. Just 'Topic 1, Topic 2'."
            f"\nText: {context}"
        )

        try:
            response = self.client.chat.completions.create(
                messages=[{'role': 'user', 'content': prompt}],
                model=self.model
            )
            
            raw_text = response.choices[0].message.content
            
            # Parse clean list
            keywords = [k.strip() for k in raw_text.split(',') if k.strip()]
            return keywords[:8] # Return top 8
            
        except Exception as e:
            print(f"AI Keyword Error: {e}")
            return []