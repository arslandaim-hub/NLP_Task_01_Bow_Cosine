import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TASK 1: Bag of Words 
corpus = [
    "The product performance is amazing and fast",
    "The service was fast and performance was great",
    "Terrible customer service and bad performance"
]

# Instantiate CountVectorizer with English stop words removed
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)

# Convert transformed matrix to DataFrame
vocab = vectorizer.get_feature_names_out()
df_bow = pd.DataFrame(X.toarray(), columns=vocab)

print("--- Task 1: Bag of Words Matrix ---")
print(df_bow, "\n")

# TASK 2: Document Search Engine & Relevance Ranking
documents = [
    "Machine learning algorithms analyze structured data effectively",
    "Deep learning and neural networks excel at processing unstructured data",
    "Natural language processing helps computers understand human language",
    "Python is widely used for machine learning and data science"
]
query = ["machine learning algorithms for data"]

vec_search = CountVectorizer()
doc_vectors = vec_search.fit_transform(documents)
query_vector = vec_search.transform(query)

# Compute cosine similarity
scores = cosine_similarity(query_vector, doc_vectors)[0]
ranked_indices = np.argsort(scores)[::-1]

print("--- Task 2: Ranked Documents ---")
for idx in ranked_indices:
    print(f"Score: {scores[idx]:.4f} | Document: {documents[idx]}")