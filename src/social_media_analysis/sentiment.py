"""TF-IDF + Logistic Regression sentiment classifier."""
from dataclasses import dataclass
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from .text_processing import clean_text


@dataclass
class SentimentModel:
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    test_accuracy: float
    confusion: object
    report: str

    def positive_probability(self, text: str) -> float:
        vec = self.vectorizer.transform([clean_text(text)])
        positive_index = list(self.classifier.classes_).index(1)
        return float(self.classifier.predict_proba(vec)[0, positive_index])


def train_sentiment_model(training_df: pd.DataFrame, max_features: int = 5000) -> SentimentModel:
    """Train/evaluate the model on a DataFrame with tweet and binary sentiment."""
    data = training_df.copy()
    data["clean"] = data["tweet"].map(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(
        data["clean"], data["sentiment"], test_size=0.2,
        random_state=42, stratify=data["sentiment"]
    )
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train_vec, y_train)
    pred = classifier.predict(X_test_vec)
    return SentimentModel(
        vectorizer=vectorizer,
        classifier=classifier,
        test_accuracy=float(accuracy_score(y_test, pred)),
        confusion=confusion_matrix(y_test, pred, labels=[0, 1]),
        report=classification_report(y_test, pred, target_names=["Negative", "Positive"], zero_division=0),
    )


def sentiment_label(score: float, low: float = 0.4, high: float = 0.6) -> str:
    """Map a positive-class probability to Positive, Negative, or Neutral."""
    if score > high:
        return "Positive"
    if score < low:
        return "Negative"
    return "Neutral"
