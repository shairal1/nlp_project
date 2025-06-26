import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
import ssl
import os


# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Download required NLTK data
'''nltk.download('punkt')
nltk.download('stopwords')'''

def read_data(file_path):
    with open(file_path, encoding="latin-1", errors="replace") as file:
        lines = file.readlines()
    
    data = []
    for line in lines:
        parts = line.strip().split('@')
        if len(parts) == 2:
            text, sentiment = parts
            data.append({'text': text.strip(), 'sentiment': sentiment.strip()})
    
    return pd.DataFrame(data)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def generate_wordcloud(text, title, filename):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def analyze_dataset(df):

    #plot the distribution of the sentiment
    plt.figure(figsize=(12, 6))
    sns.countplot(x='sentiment', data=df)
    plt.title(f'Distribution of Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.savefig(f'plots_task1/sentiment_distribution.png')
    plt.close()
 
    # Text length analysis
    df['text_length'] = df['text'].str.len()  # Using str.len() instead of apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(word_tokenize(x)))
    
    # Plot text length distribution
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='sentiment', y='text_length', data=df)
    plt.title(f'Text Length Distribution by Sentiment ')
    plt.xlabel('Sentiment')
    plt.ylabel('Text Length (characters)')
    plt.savefig(f'plots_task1/text_length.png')
    plt.close()
    
    # Plot word count distribution
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='sentiment', y='word_count', data=df)
    plt.title(f'Word Count Distribution by Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Word Count')
    plt.savefig(f'plots_task1/word_count.png')
    plt.close()
    
    # Word cloud for each sentiment
    stop_words = set(stopwords.words('english'))
    
    for sentiment in df['sentiment'].unique():
        texts = ' '.join(df[df['sentiment'] == sentiment]['text'])
        texts = preprocess_text(texts)
        words = word_tokenize(texts)
        words = [w for w in words if w not in stop_words]
        cleaned_text = ' '.join(words)
        generate_wordcloud(cleaned_text, f'Word Cloud - {sentiment}', 
                         f'plots_task1/wordcloud_{sentiment}.png')
    
    # Calculate and print unique words statistics
    print("\nUnique words per sentiment:")
    for sentiment in df['sentiment'].unique():
        texts = ' '.join(df[df['sentiment'] == sentiment]['text'])
        words = set(word_tokenize(preprocess_text(texts)))
        words = [w for w in words if w not in stop_words]
        print(f"{sentiment}: {len(words)} unique words")
    
    # Calculate average word length
    df['avg_word_length'] = df['text'].apply(lambda x: np.mean([len(word) for word in word_tokenize(x)]))
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='sentiment', y='avg_word_length', data=df)
    plt.title(f'Average Word Length by Sentiment ')
    plt.xlabel('Sentiment')
    plt.ylabel('Average Word Length')
    plt.savefig(f'plots_task1/avg_word_length.png')
    plt.close()
    
    # Most common words per sentiment
    print("\nTop 10 most common words per sentiment:")
    for sentiment in df['sentiment'].unique():
        texts = ' '.join(df[df['sentiment'] == sentiment]['text'])
        words = word_tokenize(preprocess_text(texts))
        words = [w for w in words if w not in stop_words and len(w) > 2]
        word_freq = Counter(words).most_common(10)
        print(f"\n{sentiment}:")
        for word, count in word_freq:
            print(f"{word}: {count}")
        
        # Plot the most common words per sentiment save in plots folder
        plt.figure(figsize=(12, 6))
        # Create a DataFrame for the barplot
        word_df = pd.DataFrame(word_freq, columns=['word', 'frequency'])
        sns.barplot(x='word', y='frequency', data=word_df)
        plt.title(f'Top 10 Most Common Words - {sentiment}')
        
        plt.xlabel('Word')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'plots_task1/top_10_words_{sentiment}.png')
        plt.close()

def main():
    # List of all data files
    data_files = [
        ('data/Sentences_50Agree.txt', '50% Agreement')
    ]
    all_data = []
    for file_path, agreement_level in data_files:
        df = read_data(file_path)
        all_data.append(df)
   
    combined_df = pd.concat(all_data)
    combined_df['text_length'] = combined_df['text'].str.len()
    analyze_dataset(combined_df)

if __name__ == "__main__":
    main()
