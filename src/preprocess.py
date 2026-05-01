import re
import nltk
from nltk.corpus import stopwords

# Download once
nltk.download('stopwords')

# Load stopwords
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
    Clean text for sentiment analysis while preserving useful meaning.
    Avoid over-cleaning to prevent perfect accuracy issues.
    """

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # 3. Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # 4. Keep alphabets + basic punctuation (important for sentiment)
    text = re.sub(r"[^a-zA-Z\s!?]", "", text)

    # 5. Tokenize
    words = text.split()

    # 6. Remove SOME stopwords (not all)
    # Keep important sentiment words like "not", "no"
    filtered_words = [
        word for word in words 
        if word not in STOPWORDS or word in ["not", "no"]
    ]

    # 7. Rejoin text
    cleaned_text = " ".join(filtered_words)

    return cleaned_text