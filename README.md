<div align="center">
  <h1> NLP Vector Space Modeling & Cosine Similarity</h1>
  <p><i>Implementing Bag of Words (BoW) and Search Relevance Ranking using Cosine Similarity</i></p>

  <p align="center">
    <a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python&logoColor=white" alt="Python"></a>
    <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white" alt="Scikit-Learn"></a>
    <a href="https://pandas.pydata.org/"><img src="https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white" alt="Pandas"></a>
    <a href="https://numpy.org/"><img src="https://img.shields.io/badge/numpy-013243?logo=numpy&logoColor=white" alt="NumPy"></a>
  </p>
</div>

<br>

## Overview:

This repository contains the implementation of **Natural Language Processing Lab Task 01**. The primary objective is to represent textual data as numerical vectors and compute pairwise document similarity. It demonstrates the mechanics of tokenization, vocabulary extraction, term frequency representation, and geometric distance calculations.

### Academic Details:
| **Department** | **Subject** | **Student** | **Roll Number** |
| :--- | :--- | :--- | :--- |
| Artificial Intelligence | Natural Language Processing | Arsalan Shar | 2k24/AI/15 |

---

## Tech Stack & Concepts:

* **Language:** Python 3.8+
* **Libraries:** `scikit-learn` (Feature Extraction), `pandas` (Data Structuring), `numpy` (Numerical Operations)
* **Core Concepts:** Bag of Words (BoW), Term-Frequency (TF) Matrix, Cosine Similarity, Orthogonal Vectors

---

## Screenshots:

<div align="center">
  <table>
    <tr>
      <th align="center">Task 1: Bag of Words Matrix</th>
      <th align="center">Task 2: Search Relevance Ranking</th>
    </tr>
    <tr>
      <td align="center">
        <img src="Screenshots/Screenshot 2026-09-05 164408.png" alt="BoW Matrix Output" width="400" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
      </td>
      <td align="center">
        <img src="Screenshots/Screenshot 2026-09-05 164439.png" alt="Cosine Similarity Ranking" width="400" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
      </td>
    </tr>
    <tr>
      <td align="center"><i>Figure 1: Generated sparse matrix mapping word frequencies for customer reviews.</i></td>
      <td align="center"><i>Figure 2: Documents sorted by Cosine Similarity scores against the user query.</i></td>
    </tr>
  </table>
</div>

---

## TASK Questions:

### 1. Word Order Invariance:
**Why does the sentence "Dog bites man" have the exact same Bag of Words representation as "Man bites dog"? How does this impact sentiment analysis?**.

> **Answer:**  
> The Bag of Words (BoW) model only records the *frequency* of terms in a document, discarding all syntactic context, grammar, and word order. 
> * **Representation:** Both sentences result in the exact same vector: `{"bites": 1, "dog": 1, "man": 1}`.
> * **Sentiment Impact:** Because sequence is ignored, BoW cannot capture negation or modifiers effectively. 
> * **Example:** The phrases *"good, not bad"* (positive) and *"bad, not good"* (negative) produce identical BoW vectors. A sentiment analysis model relying purely on BoW would struggle to differentiate between these opposing sentiments.

### 2. Sparsity Issue:
**What happens to the memory size and density of the BoW matrix when the corpus contains 100,000 unique vocabulary words?**.

> **Answer:**  
> The matrix becomes extremely large and highly sparse. 
> * **Memory Size:** If you have 1,000 documents, the matrix size becomes $1000 \times 100,000$ (100 million elements). 
> * **Density:** Since an average document might only contain 50 to 100 unique words, over 99.9% of the matrix will consist of zeros. The density approaches 0. 
> * **Example:** A short tweet processed against a 100,000-word vocabulary will have a vector with 10 to 20 ones and 99,980+ zeros. This wastes memory and computational power unless stored in a specialized format like a Compressed Sparse Row (CSR) matrix.

### 3. Zero Similarity:
**Explain why Document 7 in Task 2 receives a Cosine Similarity score of 0.0000 when queried against "machine learning algorithms for data".**.

> **Answer:**  
> Cosine similarity measures the cosine of the angle between two vectors using their dot product. 
> * **Reason:** Document 3 (*"Natural language processing helps computers understand human language"*) shares **zero overlapping words** with the query (*"machine learning algorithms for data"*). 
> * **Math:** When taking the dot product ($\sum A_i B_i$), multiplying the term frequencies results in zero because there are no common indices where both vectors have a non-zero value. 
> * **Example:** 
>   * Query vector for "data": `1`, Doc 3 vector for "data": `0` $\rightarrow (1 \times 0 = 0)$
>   * Doc 3 vector for "language": `2`, Query vector for "language": `0` $\rightarrow (2 \times 0 = 0)$
>   
> Because the dot product is $0$, the vectors are completely orthogonal (at a 90° angle), resulting in a final score of 0.0000.

---

## Installation & Setup:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Arslandaim-hub/NLP_Task_01.git](https://github.com/Arslandaim-hub/NLP_Task_01.git)
   cd NLP_Task_01
