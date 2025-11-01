from flask import Flask, request, jsonify
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

app = Flask(__name__)
UPLOAD_FOLDER = "uploaded_cvs"
SAMPLE_FOLDER = "backend/sample_cvs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_text_from_pdf(file_path):
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def get_recommendation(uploaded_cv_text):
    sample_texts, sample_labels = [], []
    for fname in os.listdir(SAMPLE_FOLDER):
        if fname.endswith(".pdf"):
            path = os.path.join(SAMPLE_FOLDER, fname)
            text = extract_text_from_pdf(path)
            sample_texts.append(text)
            label = fname.split(".")[0]
            sample_labels.append(label)

    all_texts = sample_texts + [uploaded_cv_text]
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(all_texts)
    similarity = cosine_similarity(vectors[-1:], vectors[:-1])
    best_match = similarity.argmax()
    return sample_labels[best_match].replace("_", " ").capitalize()

@app.route("/recommend", methods=["POST"])
def recommend():
    file = request.files["cv"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    text = extract_text_from_pdf(path)
    recommendation = get_recommendation(text)
    return jsonify({"recommendation": recommendation})

if __name__ == "__main__":
    app.run(debug=True)
