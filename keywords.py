import nltk
import re
from collections import Counter

# --- NLTK Resource Download Handler ---
# We explicitly handle the new resource names to avoid the LookupError
def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Initialize resources immediately when the module loads
setup_nltk()

# Expanded Stopwords (Targeting Data Mining/IT domain noise)
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

        # 1. Clean text slightly for tokenization
        # Replace CamelCase (e.g. "DataMining" -> "Data Mining") to help extraction
        cleaned_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        
        # 2. Tokenize using NLTK
        tokens = nltk.word_tokenize(cleaned_text)
        
        # 3. POS Tagging (Part of Speech)
        # Returns a list of tuples: [('Data', 'NNS'), ('mining', 'NN'), ...]
        # This now uses the 'averaged_perceptron_tagger_eng' resource correctly
        tagged_tokens = nltk.pos_tag(tokens)
        
        # 4. Filter for Concepts
        # We only want Nouns: NN, NNS (singular/plural nouns), NNP, NNPS (Proper nouns)
        candidate_phrases = []
        
        for word, tag in tagged_tokens:
            # Standardize
            word_lower = word.lower()
            
            # Skip stopwords and non-alpha
            if word_lower in STOPWORDS or not word.isalpha():
                continue
                
            # Keep only Nouns
            if tag.startswith('NN'):
                candidate_phrases.append(word_lower)
                
        # 5. Frequency Count
        if not candidate_phrases:
            return []

        frequency = Counter(candidate_phrases)
        
        # 6. Sort and Return Top N
        top_keywords = frequency.most_common(top_n)
        
        # Capitalize for display
        return [kw[0].capitalize() for kw in top_keywords]