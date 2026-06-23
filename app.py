from flask import Flask, jsonify, render_template, request

from model import SentimentRecommender


app = Flask(__name__)
recommender = SentimentRecommender(models_path="models")


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        users=recommender.get_users(),
        users_with_recommendations=recommender.get_users_with_recommendations(),
    )


@app.route("/recommend", methods=["POST"])
def recommend():
    user_id = request.form.get("user_id", "").strip()
    result = recommender.get_recommendations(user_id=user_id)
    return jsonify(result)


@app.route("/predict-sentiment", methods=["POST"])
def predict_sentiment():
    review_text = request.form.get("review_text", "")
    result = recommender.predict_sentiment(review_text)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
