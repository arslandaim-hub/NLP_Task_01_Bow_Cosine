import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TASK 1: Bag of Words Matrix Construction ---
# Practical Dataset: Reviews of my own application.
corpus = [
    "The video playback is extremely smooth and the UI is very intuitive.",
    "I experienced frequent crashes when trying to import data.",
    "Excellent application with no ads, highly recommend it for YOUTUBE video streaming.",
    "The battery consumption is too high when streaming media in the background.",
    "Great overall performance, but the app lacks support for custom subtitles.",
    "Terrible experience, the application freezes constantly on my device.",
    "The watch progress saving feature works perfectly every time.",
    "Fast and lightweight, but the dark mode interface needs better contrast.",
    "Customer support was incredibly helpful in resolving my database syncing issue.",
    "A solid open-source project with clean architecture and responsive design."
]

# Instantiate CountVectorizer with English stop words removed
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)

# Convert transformed matrix to DataFrame
vocab = vectorizer.get_feature_names_out()
df_bow = pd.DataFrame(X.toarray(), columns=vocab)

print("--- Task 1: Bag of Words Matrix (First 5 Rows) ---")
# Displaying first 5 rows to avoid terminal clutter with the larger vocabulary
print(df_bow.head(), "\n") 


# TASK 2: Document Search Engine & Relevance Ranking 
documents = [
    "Jetpack Compose is a modern declarative toolkit for building native UI.",
    "Machine learning algorithms analyze structured datasets to predict future trends.",
    "Natural language processing allows systems to understand and generate human language.",
    "MVVM architecture separates the user interface from the underlying business logic.",
    "Deep learning relies on artificial neural networks to process unstructured data.",
    "Retrieval-augmented generation improves language models using external documents.",
    "Kotlin coroutines provide a simplified way to manage asynchronous tasks.",
    "Cosine similarity calculates the geometric angle between two text vectors.",
    "Using local databases ensures offline data persistence for mobile applications.",
    "Data science involves extracting insights from noisy data using statistical modeling."
]

# A more complex query to test varying similarity scores
query = ["machine learning and natural language processing for unstructured data"]

vec_search = CountVectorizer(stop_words='english')
doc_vectors = vec_search.fit_transform(documents)
query_vector = vec_search.transform(query)

# Compute cosine similarity
scores = cosine_similarity(query_vector, doc_vectors)[0]
ranked_indices = np.argsort(scores)[::-1]

print("--- Task 2: Ranked Documents ---")
for idx in ranked_indices:
    # Only print documents that have at least some relevance, or print all to see the 0.0 scores
    print(f"Score: {scores[idx]:.4f} | Document: {documents[idx]}")