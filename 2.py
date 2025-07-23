from flask import Flask, request, jsonify
import pickle

# Load the saved model and vectorizer
with open('document_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Define category mapping
CATEGORY_MAPPING = {
    1: "Science",
    2: "Technology",
    3: "Business",
    4: "Entertainment"
}

# Initialize Flask app
app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify_document():
    try:
        # Get the text from the request
        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Preprocess and vectorize the text
        text_vectorized = vectorizer.transform([text])

        # Predict the category
        prediction = model.predict(text_vectorized)
        category_name = CATEGORY_MAPPING.get(int(prediction[0]), "Unknown")

        return jsonify({"category": category_name}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the server
if __name__ == '__main__':
    app.run(debug=True)
