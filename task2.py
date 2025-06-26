import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import contractions
import unicodedata

# Download required NLTK data
'''nltk.download('punkt')
nltk.download('stopwords')'''
nltk.download('wordnet')

def read_data(file_path):
    """Read the financial news dataset."""
    with open(file_path, encoding="latin-1", errors="replace") as file:
        lines = file.readlines()
    
    data = []
    for line in lines:
        parts = line.strip().split('@')
        if len(parts) == 2:
            text, sentiment = parts
            data.append({'text': text.strip(), 'sentiment': sentiment.strip()})
    
    return pd.DataFrame(data)

def remove_special_characters(text):
    """Remove special characters and digits while preserving spaces."""
    # Keep only letters, spaces, and basic punctuation
    text = re.sub(r'[^a-zA-Z\s.,!?]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def expand_contractions(text):
    """Expand contractions in text."""
    return contractions.fix(text)

def remove_punctuation(text):
    """Remove punctuation from text."""
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_numbers(text):
    """Remove numbers from text."""
    return re.sub(r'\d+', '', text)

def remove_extra_spaces(text):
    """Remove extra spaces from text."""
    return re.sub(r'\s+', ' ', text).strip()

def remove_accented_chars(text):
    """Remove accented characters from text."""
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')

def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    """
    Apply all pre-processing steps to the text.
    
    Args:
        text (str): Input text
        remove_stopwords (bool): Whether to remove stopwords
        lemmatize (bool): Whether to lemmatize words
    
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Expand contractions
    text = expand_contractions(text)
    
    # Remove accented characters
    text = remove_accented_chars(text)
    
    # Remove special characters and numbers
    text = remove_special_characters(text)
    text = remove_numbers(text)
    
    # Remove punctuation
    text = remove_punctuation(text)
    
    # Remove extra spaces
    text = remove_extra_spaces(text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    if remove_stopwords:
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    
    if lemmatize:
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def main():
    # Read the data
    data_files = [
        ('data/Sentences_50Agree.txt', '50% Agreement'),
        #('data/Sentences_66Agree.txt', '66% Agreement'),
        #('data/Sentences_75Agree.txt', '75% Agreement'),
        #('data/Sentences_AllAgree.txt', '100% Agreement')
    ]
    df = []
    for file_path, agreement_level in data_files:
        df_parts = read_data(file_path)
        df.append(df_parts)
    df = pd.concat(df)
    
    # Create a copy of the original text
    df['original_text'] = df['text']
    
    # Apply pre-processing
    print("Applying pre-processing steps...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Print some examples
    print("\nExample of pre-processing results:")
    print("\nOriginal text:")
    print(df['original_text'].iloc[0])
    print("\nProcessed text:")
    print(df['processed_text'].iloc[0])
    
    # Save the processed data
    df.to_csv('processed_data.csv', index=False)
    print("\nProcessed data saved to 'processed_data.csv'")
    
    # Print some statistics
    print("\nPre-processing Statistics:")
    print(f"Total number of sentences: {len(df)}")
    print("\nAverage text length before processing:", df['original_text'].str.len().mean())
    print("Average text length after processing:", df['processed_text'].str.len().mean())
    
    # Count unique words before and after processing
    original_words = set(' '.join(df['original_text']).lower().split())
    processed_words = set(' '.join(df['processed_text']).lower().split())
    print("\nNumber of unique words before processing:", len(original_words))
    print("Number of unique words after processing:", len(processed_words))

if __name__ == "__main__":
    main() 