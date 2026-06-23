from pathlib import Path
import html
import pickle
import re
import unicodedata

import numpy as np
import pandas as pd


class SentimentRecommender:
    """
    Service class for the sentiment-based product recommendation project.

    Responsibilities:
    - Load saved recommendation and sentiment artifacts.
    - Recommend top products for a selected user.
    - Predict sentiment for manually entered review text.
    """

    sentiment_model_file = "top_sentiment_classifier_model.pkl"
    tfidf_vectorizer_file = "tfidf_vectorizer.pkl"
    recommendation_matrix_file = "final_recommendation_model.pkl"
    cleaned_data_file = "cleansed_data.pkl"

    negative_review_patterns = [
        r"\bdo\s+not\s+like\b",
        r"\bdon\s*t\s+like\b",
        r"\bdont\s+like\b",
        r"\bnot\s+good\b",
        r"\bnot\s+happy\b",
        r"\bnot\s+satisfied\b",
        r"\bnot\s+recommend\b",
        r"\bwould\s+not\s+recommend\b",
        r"\bwill\s+not\s+buy\b",
        r"\bnever\s+buy\b",
        r"\bhate\b",
        r"\bterrible\b",
        r"\bawful\b",
        r"\bworst\b",
        r"\bbad\b",
        r"\bpoor\b",
        r"\bdisappointed\b",
        r"\buseless\b",
        r"\bwaste\b",
        r"\breturned\b",
    ]

    def __init__(self, models_path="models"):
        self.models_path = Path(models_path)

        self.sentiment_model = self._load_artifact(self.sentiment_model_file)
        self.tfidf_vectorizer = self._load_artifact(self.tfidf_vectorizer_file)
        self.recommendation_matrix = self._load_artifact(self.recommendation_matrix_file)
        self.cleaned_data = self._load_artifact(self.cleaned_data_file)

        self._validate_artifacts()

    def _load_artifact(self, file_name):
        artifact_path = self.models_path / file_name

        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

        with artifact_path.open("rb") as file:
            artifact = pickle.load(file)

        if isinstance(artifact, dict) and "model" in artifact:
            return artifact["model"]

        return artifact

    def _validate_artifacts(self):
        if not isinstance(self.recommendation_matrix, pd.DataFrame):
            raise TypeError("final_recommendation_model.pkl must contain a pandas DataFrame.")

        if not isinstance(self.cleaned_data, pd.DataFrame):
            raise TypeError("cleansed_data.pkl must contain a pandas DataFrame.")

        required_columns = {"id", "name", "lemmatized_review_text"}
        missing_columns = required_columns.difference(self.cleaned_data.columns)

        if missing_columns:
            raise ValueError(
                f"cleansed_data.pkl is missing required columns: {sorted(missing_columns)}"
            )

    def get_users(self):
        return sorted(self.recommendation_matrix.index.astype(str).tolist())

    def _resolve_user_id(self, user_id):
        if user_id in self.recommendation_matrix.index:
            return user_id

        user_id_as_string = str(user_id)
        user_lookup = {
            str(index_value): index_value
            for index_value in self.recommendation_matrix.index
        }

        return user_lookup.get(user_id_as_string)

    def get_recommendations(self, user_id, top_k=5, candidate_count=20):
        resolved_user_id = self._resolve_user_id(user_id)

        if resolved_user_id is None:
            return {
                "status": "error",
                "message": f"User '{user_id}' was not found.",
                "recommendations": [],
            }

        user_scores = (
            self.recommendation_matrix
            .loc[resolved_user_id]
            .sort_values(ascending=False)
        )

        positive_user_scores = user_scores[user_scores > 0]

        if positive_user_scores.empty:
            return {
                "status": "empty",
                "message": (
                    f"No strong recommendations are available for user '{user_id}'. "
                    "This usually happens when the user has too few ratings or no similar "
                    "users with overlapping product preferences."
                ),
                "recommendations": [],
            }

        candidate_product_ids = (
            positive_user_scores
            .head(candidate_count)
            .index
            .tolist()
        )

        candidate_reviews = self.cleaned_data[
            self.cleaned_data["id"].isin(candidate_product_ids)
        ].copy()

        if candidate_reviews.empty:
            return {
                "status": "empty",
                "message": "No review records were found for the recommended products.",
                "recommendations": [],
            }

        review_text = candidate_reviews["lemmatized_review_text"].fillna("").astype(str)
        review_features = self.tfidf_vectorizer.transform(review_text)
        predictions = self.sentiment_model.predict(review_features)

        candidate_reviews["predicted_sentiment"] = predictions
        candidate_reviews["positive_sentiment"] = np.where(
            candidate_reviews["predicted_sentiment"].astype(str).str.lower().str.startswith("pos"),
            1,
            0,
        )

        sentiment_summary = (
            candidate_reviews
            .groupby(["id", "name"], as_index=False)
            .agg(
                positive_sentiment_count=("positive_sentiment", "sum"),
                total_review_count=("predicted_sentiment", "count"),
            )
        )

        sentiment_summary["positive_sentiment_percentage"] = (
            sentiment_summary["positive_sentiment_count"]
            .div(sentiment_summary["total_review_count"])
            .mul(100)
            .round(2)
        )

        recommendation_scores = (
            positive_user_scores
            .loc[candidate_product_ids]
            .rename("recommendation_score")
            .reset_index()
            .rename(columns={"index": "id"})
        )

        final_recommendations = (
            sentiment_summary
            .merge(recommendation_scores, on="id", how="left")
            .sort_values(
                by=[
                    "positive_sentiment_percentage",
                    "recommendation_score",
                    "total_review_count",
                ],
                ascending=False,
            )
            .head(top_k)
        )

        recommendations = []

        for rank, (_, row) in enumerate(final_recommendations.iterrows(), start=1):
            recommendations.append({
                "rank": rank,
                "product_id": row["id"],
                "product_name": row["name"],
                "positive_sentiment_count": int(row["positive_sentiment_count"]),
                "total_review_count": int(row["total_review_count"]),
                "positive_sentiment_percentage": float(row["positive_sentiment_percentage"]),
                "recommendation_score": float(row["recommendation_score"]),
            })

        return {
            "status": "success",
            "message": f"Top {len(recommendations)} products recommended for user '{user_id}'.",
            "recommendations": recommendations,
        }

    def _clean_review_text(self, review_text):
        text = html.unescape(str(review_text))
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
        text = (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )
        text = text.lower()

        contraction_rules = {
            r"\bcan't\b": "cannot",
            r"\bwon't\b": "will not",
            r"\bain't\b": "is not",
            r"n't\b": " not",
            r"'re\b": " are",
            r"'ve\b": " have",
            r"'ll\b": " will",
            r"'m\b": " am",
            r"'d\b": " would",
            r"'s\b": " is",
        }

        for pattern, replacement in contraction_rules.items():
            text = re.sub(pattern, replacement, text)

        text = re.sub(r"\b\w*\d+\w*\b", " ", text)
        text = re.sub(r"\b\w{25,}\b", " ", text)
        text = re.sub(r"([a-z])\1{2,}", r"\1\1", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _has_explicit_negative_sentiment(self, review_text):
        normalized_text = self._clean_review_text(review_text)

        return any(
            re.search(pattern, normalized_text)
            for pattern in self.negative_review_patterns
        )

    def predict_sentiment(self, review_text):
        cleaned_review_text = str(review_text).strip()

        if not cleaned_review_text:
            return {
                "status": "error",
                "message": "Please enter review text before predicting sentiment.",
                "prediction": None,
            }

        model_input = self._clean_review_text(cleaned_review_text)
        review_features = self.tfidf_vectorizer.transform([model_input])
        predicted_sentiment = self.sentiment_model.predict(review_features)[0]

        confidence = None
        probabilities = {}

        if hasattr(self.sentiment_model, "predict_proba"):
            probability_values = self.sentiment_model.predict_proba(review_features)[0]
            probabilities = {
                str(label): float(round(probability * 100, 2))
                for label, probability in zip(self.sentiment_model.classes_, probability_values)
            }
            confidence = probabilities.get(str(predicted_sentiment))

        used_rule_override = False

        if self._has_explicit_negative_sentiment(cleaned_review_text):
            predicted_sentiment = "Negative"
            confidence = 90.0
            probabilities = {
                "Negative": 90.0,
                "Positive": 10.0,
            }
            used_rule_override = True

        return {
            "status": "success",
            "message": "Sentiment predicted successfully.",
            "prediction": {
                "review_text": cleaned_review_text,
                "model_input": model_input,
                "sentiment": str(predicted_sentiment),
                "confidence": confidence,
                "probabilities": probabilities,
                "used_rule_override": used_rule_override,
            },
        }
