import os
import joblib
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from preprocess import preprocess_text
# Download NLTK resources 
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load CSV
df = pd.read_csv("annotations1.csv")

# Initialize preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()



# Apply preprocessing
for col in ['Generation_prompt', 'Annotation1', 'Annotation2', 'Annotation3']:
    df[f'{col}_clean'] = df[col].apply(preprocess_text)

# Check cleaned results
print(df[['Generation_prompt_clean', 'Annotation1_clean', 'Annotation2_clean', 'Annotation3_clean']].head())
df.to_csv("annotations_cleaned.csv", index=False)

prompts = df['Generation_prompt_clean']
annotations = df[['Annotation1_clean', 'Annotation2_clean', 'Annotation3_clean']].apply(lambda x: ' '.join(x), axis=1)

# TF-IDF for prompts and annotations (all words)
prompt_vectorizer = TfidfVectorizer()
prompt_tfidf = prompt_vectorizer.fit_transform(prompts)
prompt_keywords = prompt_vectorizer.get_feature_names_out()

annotation_vectorizer = TfidfVectorizer()
annotation_tfidf = annotation_vectorizer.fit_transform(annotations)
annotation_keywords = annotation_vectorizer.get_feature_names_out()


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a list of all rows: prompt + annotations per image
tfidf_corpus = []

for idx, row in df.iterrows():
    tfidf_corpus.append(row['Generation_prompt_clean'])
    tfidf_corpus.append(row['Annotation1_clean'])
    tfidf_corpus.append(row['Annotation2_clean'])
    tfidf_corpus.append(row['Annotation3_clean'])

# keep track of what type each row 
row_labels = []
for idx in range(len(df)):
    row_labels.extend([
        f"{idx}_Generation_prompt",
        f"{idx}_Annotation1",
        f"{idx}_Annotation2",
        f"{idx}_Annotation3"
    ])

# Create TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(tfidf_corpus)

# Convert to DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=row_labels)

#LSA
from sklearn.decomposition import TruncatedSVD


n_topics = 15  # number of latent concepts/topics
lsa = TruncatedSVD(n_components=n_topics, random_state=42)
lsa_matrix = lsa.fit_transform(tfidf_matrix)



terms = tfidf_df.columns  # TF-IDF feature names

for i, comp in enumerate(lsa.components_):
    terms_in_topic = [terms[idx] for idx in comp.argsort()[-10:][::-1]]  # top 10 words
    print(f"Topic {i+1}: {terms_in_topic}")

#cosine similarities
from sklearn.metrics.pairwise import cosine_similarity

# Pick the rows
vec_prompt = lsa_matrix[0].reshape(1, -1)      # 0_prompt
vec_ann2 = lsa_matrix[2].reshape(1, -1)       # 0_annotation2
vec_other = lsa_matrix[14].reshape(1, -1)     # 15th row (unrelated)

# Cosine similarity
sim_prompt_ann2 = cosine_similarity(vec_prompt, vec_ann2)[0][0]
sim_prompt_other = cosine_similarity(vec_prompt, vec_other)[0][0]

print(f"Cosine similarity (prompt vs Annotation2): {sim_prompt_ann2:.4f}")
print(f"Cosine similarity (prompt vs unrelated 15th row): {sim_prompt_other:.4f}")


import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# First row vector (first image's prompt)
vec_prompt = lsa_matrix[4].reshape(1, -1)

# Compute cosine similarity with all 120 rows
cos_sim_all = cosine_similarity(vec_prompt, lsa_matrix)[0]

# Convert to DataFrame for easier plotting
cos_sim_df = pd.DataFrame({
    'row': row_labels,
    'cosine_similarity': cos_sim_all
})


#get the mean of annotations 
num_images = lsa_matrix.shape[0] // 4
mean_vectors = []

for i in range(num_images):
    block = lsa_matrix[4*i : 4*i + 4]       # rows for image i
    mean_vec = block.mean(axis=0)  # average of 4 rows
    mean_vectors.append(mean_vec)

mean_vectors = np.vstack(mean_vectors)
mean_df = pd.DataFrame(mean_vectors)

#load COCO annotations
import json

with open("annotations/captions_val2014.json", "r") as f:
    coco_data = json.load(f)

# Extract captions and corresponding image IDs
list_of_coco_captions = []
coco_image_ids = []

for ann in coco_data['annotations']:
    list_of_coco_captions.append(ann['caption'])
    coco_image_ids.append(ann['image_id'])

#TFIDF and LSA for COCO
coco_tfidf = vectorizer.transform(list_of_coco_captions)
coco_lsa = lsa.transform(coco_tfidf)

#apply cosine similarity to COCO dataset
from sklearn.metrics.pairwise import cosine_similarity

top_k = 4  # top-4 most similar
retrieved_results = []

for i, row in mean_df.iterrows():
    img_vec = row.values.reshape(1, -1)      # shape (1, n_topics)
    my_vec = img_vec.flatten()               # shape (n_topics,)

    # cosine similarity with all COCO captions
    sims = cosine_similarity(img_vec, coco_lsa)[0]   # shape: (num_captions,)

    # get top-k indices by cosine similarity
    top_idx = sims.argsort()[::-1][:top_k]

    # build results: (cosine_sim, euclidean_dist, coco_image_id, coco_caption_index)
    img_results = []
    for j in top_idx:
        coco_vec = coco_lsa[j]
        eucl_dist = np.linalg.norm(my_vec - coco_vec)
        img_results.append((sims[j], eucl_dist, coco_image_ids[j], j))

    retrieved_results.append(img_results)

#part to show results
image_index = 0  
results = retrieved_results[image_index]

print(f"Top-{len(results)} retrieved COCO matches for image {image_index}:\n")

for rank, (cos_sim, euclid_dist, image_id, caption_idx) in enumerate(results, start=1):
    print(f"[Rank {rank}]")
    print(f"  COCO image ID     : {image_id}")
    print(f"  Caption index     : {caption_idx}")
    print(f"  Cosine similarity : {cos_sim:.4f}")
    print(f"  Euclidean dist    : {euclid_dist:.4f}")

    image_path = f"train2014/COCO_train2014_{image_id:012d}.jpg"
    print(f"  Path              : {image_path}\n")


os.makedirs("data", exist_ok=True)

np.save("data/mean_vectors.npy", mean_vectors)
np.save("data/coco_lsa.npy", coco_lsa)

with open("data/coco_image_ids.json", "w") as f:
    json.dump(coco_image_ids, f)

joblib.dump(vectorizer, "data/tfidf.pkl")
joblib.dump(lsa, "data/lsa.pkl")

print("Precomputation finished and saved.")