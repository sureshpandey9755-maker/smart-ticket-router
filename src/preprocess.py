import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans raw ticket text by lowercasing, removing punctuation, 
    removing stop words, and applying lemmatization.
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize words
    words = text.split()
    
    # Remove stop words and lemmatize
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    
    return ' '.join(cleaned_words)