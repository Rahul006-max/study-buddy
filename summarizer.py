import os
from groq import Groq

class GroqSummarizer:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.model = 'llama3-70b-8192' # Or use 'llama3-8b-8192' for speed

    def summarize(self, text):
        if not text:
            return "No text to summarize."

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert academic assistant. Summarize the text comprehensively covering all key points."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                model=self.model,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"AI Error: {str(e)}"