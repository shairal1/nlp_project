import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def read_data(file_path):
    """Read the processed data."""
    return pd.read_csv(file_path)

def create_pipeline_baysen():
    """Create a pipeline for text classification."""
    return ImbPipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )),
        ('smote', SMOTE(random_state=42)),
        ('classifier', MultinomialNB())
    ])

def create_pipeline_logistic_ann():
    """Create a pipeline for logistic regression text classification."""
    return ImbPipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )),
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

def plot_confusion_matrix(y_true, y_pred, labels, model_name):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix – {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if model_name == 'baysen':
        plt.savefig('plots/NB_tfidf_confusion_matrix.png')
    elif model_name == 'logistic_ann':
        plt.savefig('plots/LR_tfidf_confusion_matrix.png')
    else:
        plt.savefig(f'plots/confusion_matrix_{model_name}.png')
    
    plt.close()

def plot_classification_report(y_true, y_pred, labels, model_name):
    """Plot and print classification report."""
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    print(f"\nClassification Report – {model_name}")
    print(classification_report(y_true, y_pred, target_names=labels))

    report_df = pd.DataFrame(report).transpose().drop('support', axis=1)
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df, annot=True, cmap='Blues', fmt='.2f')
    plt.title(f'Classification Report – {model_name}')
    plt.tight_layout()

    if model_name == 'baysen':
        plt.savefig('plots/NB_tfidf_classification_report.png')
    elif model_name == 'logistic_ann':
        plt.savefig('plots/LR_tfidf_classification_report.png')
    else:
        plt.savefig(f'plots/classification_report_{model_name}.png')
    
    plt.close()


def main(model_name):
    """Main function to train and evaluate the model."""
    print(f"\n=== Running model: {model_name} ===")

    df = read_data('processed_data.csv').dropna()
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['sentiment'] = df['sentiment'].map(sentiment_map)

    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['sentiment'],
        test_size=0.2, random_state=42, stratify=df['sentiment']
    )

    if model_name == 'logistic_ann':
        pipeline = create_pipeline_logistic_ann()
    elif model_name == 'baysen':
        pipeline = create_pipeline_baysen()
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    labels = ['negative', 'neutral', 'positive']
    plot_confusion_matrix(y_test, y_pred, labels, model_name)
    plot_classification_report(y_test, y_pred, labels, model_name)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy – {model_name}: {acc:.3f}")

# Run both models
main('logistic_ann')
main('baysen')