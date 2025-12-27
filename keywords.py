import nltk
import re
import os
from collections import Counter

# --- NLTK Cloud Setup ---
# On Streamlit Cloud, we need to explicitly point to a writable directory
# to store the NLTK data.
nltk_data_dir = os.path.expanduser('~/nltk_data')
# Add this directory to NLTK's search path
nltk.data.path.insert(0, nltk_data_dir)

def setup_nltk():
    """Downloads necessary NLTK data for the cloud environment."""
    # 1. Download the new Tab-based Tokenizer (Required for Python 3.12+ / New NLTK)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        # Download to our specific directory
        nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
    
    # 2. Download the POS Tagger
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_dir, quiet=True)

# Run setup immediately
setup_nltk()

# --- Stopwords ---
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
    'also', 'use', 'used', 'using', 'data', 'information', 'source', 'type', 'often',
    'may', 'one', 'two', 'first', 'second', 'within', 'include', 'available', 'provide'
}

class ConceptExtractor:
    def __init__(self):
        pass

    def extract_keywords(self, text, top_n=10):
        if not text:
            return []

        # 1. Clean text (Split CamelCase)
        cleaned_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        
        # 2. Tokenize
        tokens = nltk.word_tokenize(cleaned_text)
        
        # 3. POS Tagging
        tagged_tokens = nltk.pos_tag(tokens)
        
        # 4. Filter Nouns
        candidate_phrases = []
        
        for word, tag in tagged_tokens:
            word_lower = word.lower()
            if word_lower in STOPWORDS or not word.isalpha():
                continue
            if tag.startswith('NN'):
                candidate_phrases.append(word_lower)
                
        if not candidate_phrases:
            return []

        # 5. Frequency
        frequency = Counter(candidate_phrases)
        top_keywords = frequency.most_common(top_n)
        
        return [kw[0].capitalize() for kw in top_keywords]