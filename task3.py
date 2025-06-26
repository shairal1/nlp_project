#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

import joblib

def read_data(file_path):
    """Read the processed data and encode sentiments."""
    df = pd.read_csv(file_path)
    df = df.dropna()
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['sentiment'] = df['sentiment'].map(sentiment_map)
    return df

def vectorize_data(X_train, X_test, method="tfidf"):
    """Vectorize text using TF-IDF or CountVectorizer."""
    if method == "tfidf":
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95)
    elif method == "count":
        vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95)
    else:
        raise ValueError("Vektorisierungsmethode nicht erkannt.")
    return vectorizer, vectorizer.fit_transform(X_train), vectorizer.transform(X_test)


def apply_smote(X_train_vec, y_train):
    """Balance classes using SMOTE."""
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train_vec, y_train)

def evaluate_model(model, X_test, y_test, labels, name):
    """Evaluate model and save confusion matrix and classification report plots_task3."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"\n{name} - Accuracy: {acc:.3f}, F1: {f1:.3f}")
    print(classification_report(y_test, y_pred, target_names=labels))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title(f'Confusion Matrix – {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'plots_task3/{name}_confusion_matrix.png')
    plt.close()

    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose().drop('support', axis=1)
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df, annot=True, cmap='Blues', fmt='.2f')
    plt.title(f'Classification Report – {name}')
    plt.tight_layout()
    plt.savefig(f'plots_task3/{name}_classification_report.png')
    plt.close()


def train_naive_bayes(X_train, y_train):
    """Train Naive Bayes model."""
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    return clf


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model."""
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf


def train_ffnn(X_train, y_train, X_test, y_test):
    """Train a Feedforward Neural Network (FFNN)."""
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)

    model = Sequential()
    model.add(Dense(512, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train.toarray(), y_train_cat, epochs=10, batch_size=64, validation_split=0.2, callbacks=[es], verbose=2)

    y_pred = np.argmax(model.predict(X_test.toarray()), axis=1)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"\nFeedforward Neural Network - F1 Score: {f1:.3f}")
    print(classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']))

    # Save FFNN evaluation plots_task3
    labels = ['negative', 'neutral', 'positive']
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title(f'Confusion Matrix – FFNN')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('plots_task3/FFNN_confusion_matrix.png')
    plt.close()

    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose().drop('support', axis=1)
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df, annot=True, cmap='Blues', fmt='.2f')
    plt.title(f'Classification Report – FFNN')
    plt.tight_layout()
    plt.savefig('plots_task3/FFNN_classification_report.png')
    plt.close()

    return model


def main():
    """Main function to run sentiment classification with various models."""
    df = read_data("processed_data.csv")
    labels = ['negative', 'neutral', 'positive']
    X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['sentiment'], test_size=0.2, stratify=df['sentiment'], random_state=42)

    for method in ["tfidf", "count"]:
        print(f"\n--- Vectorizer: {method.upper()} ---")
        vectorizer, X_train_vec, X_test_vec = vectorize_data(X_train, X_test, method)
        X_train_resampled, y_train_resampled = apply_smote(X_train_vec, y_train)

        nb_model = train_naive_bayes(X_train_resampled, y_train_resampled)
        evaluate_model(nb_model, X_test_vec, y_test, labels, f"NB_{method}")

        lr_model = train_logistic_regression(X_train_resampled, y_train_resampled)
        evaluate_model(lr_model, X_test_vec, y_test, labels, f"LR_{method}")

        if method == "tfidf":
            print("\nTraining FFNN with TF-IDF")
            ffnn_model = train_ffnn(X_train_resampled, y_train_resampled, X_test_vec, y_test)

    print("\n--- Binary Classification (Positive vs Negative) ---")
    df_bin = df[df['sentiment'] != 1]
    Xb = df_bin['processed_text']
    yb = df_bin['sentiment']
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb, test_size=0.2, stratify=yb, random_state=42)
    vec, Xb_train_vec, Xb_test_vec = vectorize_data(Xb_train, Xb_test, "tfidf")
    Xb_train_sm, yb_train_sm = apply_smote(Xb_train_vec, yb_train)
    bin_model = train_logistic_regression(Xb_train_sm, yb_train_sm)
    evaluate_model(bin_model, Xb_test_vec, yb_test, ['negative', 'positive'], "Binary_LogReg")


if __name__ == "__main__":
    main()
