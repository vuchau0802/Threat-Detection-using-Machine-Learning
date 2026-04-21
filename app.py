from flask import Flask, render_template, request, jsonify
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load stopwords and models
with open("stopwords.txt", "r") as file:
    stopwords = file.read().splitlines()

vocab = pickle.load(open("tfidfvectoizer.pkl", "rb"))
vectorizer = TfidfVectorizer(stop_words=stopwords, lowercase=True, vocabulary=vocab)
model = pickle.load(open("LinearSVCTuned.pkl", 'rb'))

# Optional: Simple keyword list to highlight bullying words
bullying_keywords = ["idiot", "stupid", "dumb", "ugly", "hate", "loser", "kill", "fat", "die", "freak"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_input = data.get('text', '')

    if not user_input.strip():
        return jsonify({"error": "Empty input"})

    transformed_input = vectorizer.fit_transform([user_input])
    prediction = int(model.predict(transformed_input)[0])
    decision_score = model.decision_function(transformed_input)[0]
    confidence = round(min(max((abs(decision_score) / 5.0) * 100, 50), 100), 2)  # crude confidence

    # Simple sentiment logic (demo purposes)
    sentiment = "Neutral"
    if any(word in user_input.lower() for word in ["happy", "good", "love", "great", "nice"]):
        sentiment = "Positive"
    elif any(word in user_input.lower() for word in ["bad", "sad", "angry", "hate", "terrible"]):
        sentiment = "Negative"

    # Highlight bullying words
    highlighted = user_input
    for word in bullying_keywords:
        highlighted = re.sub(fr'\b({word})\b', r'<mark class="bg-danger text-white">\1</mark>', highlighted, flags=re.IGNORECASE)

    return jsonify({
        'prediction': prediction,
        'confidence': confidence,
        'highlighted': highlighted,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
