import random

import nltk
from nltk.corpus import movie_reviews, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def preprocess_text(text, stop_words, lemmatizer):
    """
    Предобработка текста: токенизация, удаление стоп-слов и пунктуации,
    лемматизация и объединение в строку.

    Args:
        text (str): исходный текст.
        stop_words (set): множество стоп-слов.
        lemmatizer (WordNetLemmatizer): объект лемматизатора.

    Returns:
        str: обработанный текст в виде строки лемматизированных токенов.
    """
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return ' '.join(lemmatized_tokens)


def main():
    """
    Основная функция: загрузка данных, предобработка, обучение модели,
    оценка точности и вывод примеров предсказаний.
    """
    nltk.download('movie_reviews')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    documents = [(movie_reviews.raw(fileid), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]

    random.shuffle(documents)

    texts = [preprocess_text(text, stop_words, lemmatizer) for text, _ in documents]
    labels = [label for _, label in documents]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42)

    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy модели: {accuracy:.4f}")

    print("\nПримеры предсказаний:")
    indices = random.sample(range(len(X_test)), 3)
    for i in indices:
        print(f"\nТекст (предобработанный): {X_test[i]}")
        print(f"Истинный класс: {y_test[i]}")
        print(f"Предсказанный класс: {y_pred[i]}")


if __name__ == "__main__":
    main()
