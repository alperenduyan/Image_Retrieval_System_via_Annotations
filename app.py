from flask import Flask, render_template, request, jsonify, url_for
import numpy as np
import json
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import preprocess_text
import os

app = Flask(__name__)

# Load precomputed data
mean_vectors = np.load("data/mean_vectors.npy")
coco_lsa = np.load("data/coco_lsa.npy")

with open("data/coco_image_ids.json") as f:
    coco_image_ids = json.load(f)

vectorizer = joblib.load("data/tfidf.pkl")
lsa = joblib.load("data/lsa.pkl")

TOP_K = 4

@app.route("/")
def index():
    # List only numeric jpg files
    image_files = [
        f for f in os.listdir("static/my_images")
        if f.endswith(".jpg") and f.split(".")[0].isdigit()
    ]

    # Sort numerically
    image_files = sorted(image_files, key=lambda x: int(x.split(".")[0]))

    
    images = []
    for img in image_files:
        img_idx = int(img.split(".")[0]) - 1  # list start from 0 but images start from 1
        images.append((url_for('static', filename=f"my_images/{img}"), img_idx))

    return render_template("index.html", images=images)


@app.route("/similar/image/<int:image_id>")
def similar_image(image_id):
    query_vec = mean_vectors[image_id].reshape(1, -1)
    sims = cosine_similarity(query_vec, coco_lsa)[0]

    top_idx = sims.argsort()[::-1][:TOP_K]

    results = []
    for j in top_idx:
        results.append({
            "path": f"static/coco/val2014/COCO_val2014_{coco_image_ids[j]:012d}.jpg",
            "score": float(sims[j])
        })

    return jsonify(results)

@app.route("/search", methods=["POST"])
def search():
    query = request.form["query"]
    clean = preprocess_text(query)

    tfidf = vectorizer.transform([clean])
    query_lsa = lsa.transform(tfidf)

    sims = cosine_similarity(query_lsa, coco_lsa)[0]
    top_idx = sims.argsort()[::-1][:TOP_K]

    results = []
    for j in top_idx:
        results.append({
            "path": f"static/coco/val2014/COCO_val2014_{coco_image_ids[j]:012d}.jpg",
            "score": float(sims[j])
        })

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
