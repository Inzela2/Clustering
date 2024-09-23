import os
import re
import pickle
import numpy as np
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Define the directory where the text files are stored
data_dir = 'C:/Users/asus/Downloads/clustering/archive (3)'  # Update this to the correct path

# Get all file names from the directory
file_names = [f for f in os.listdir(data_dir) if f.endswith('.txt')]

# Load the content from all 20 files
documents = []
for file_name in file_names:
    with open(os.path.join(data_dir, file_name), 'r', encoding='utf-8', errors='ignore') as file:
        documents.append(file.read())


# Define preprocessing steps using a custom approach
def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    # Removing punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Removing numbers
    text = re.sub(r'\d+', '', text)

    return text


# Apply preprocessing to all documents
documents = [preprocess_text(doc) for doc in documents]

# Preprocess: Define stopwords and initialize the vectorizer
stop_words = stopwords.words('english')
vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=0.5, min_df=2)

# Convert documents into TF-IDF features
X_tfidf = vectorizer.fit_transform(documents)

# Paths to save/load TF-IDF embeddings
pkl_path = 'C:/Users/asus/Downloads/clustering/tfidf_embeddings.pkl'
txt_path = 'C:/Users/asus/Downloads/clustering/tfidf_embeddings.txt'

# Save the TF-IDF embeddings to a .pkl file
with open(pkl_path, 'wb') as f:
    pickle.dump(X_tfidf, f)
print(f"TF-IDF embeddings saved to {pkl_path}")

# Also save the embeddings to a .txt file for inspection
with open(txt_path, 'w') as f:
    for i in range(X_tfidf.shape[0]):
        f.write(f"Document {i + 1} Embedding:\n")
        np.savetxt(f, X_tfidf[i].toarray(), delimiter=',', fmt='%f')
        f.write("\n")
print(f"TF-IDF embeddings also saved as text at {txt_path}")

# Reduce dimensionality using TruncatedSVD
svd = TruncatedSVD(n_components=2, random_state=42)
X_reduced = svd.fit_transform(X_tfidf)

# Output directory for images
output_dir = 'C:/Users/asus/Downloads/clustering'


def evaluate_clustering(true_labels, predicted_labels, method_name):
    """
    Evaluate clustering results using multiple metrics
    """
    sil_score = silhouette_score(X_reduced, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    ari = adjusted_rand_score(true_labels, predicted_labels)
    fmi = fowlkes_mallows_score(true_labels, predicted_labels)

    print(f"\n{method_name} Evaluation Metrics:")
    print(f"Silhouette Score: {sil_score}")
    print(f"Normalized Mutual Information (NMI): {nmi}")
    print(f"Adjusted Rand Index (ARI): {ari}")
    print(f"Fowlkes-Mallows Index (FMI): {fmi}")

    return sil_score, nmi, ari, fmi


# Since we don't have true labels, I'll use cluster assignments from KMeans as a pseudo ground truth example.
pseudo_true_labels = KMeans(n_clusters=3, random_state=42).fit_predict(X_reduced)

# 1. KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters_kmeans = kmeans.fit_predict(X_reduced)

# Evaluate KMeans
sil_score_kmeans, nmi_kmeans, ari_kmeans, fmi_kmeans = evaluate_clustering(pseudo_true_labels, clusters_kmeans,
                                                                           "KMeans")

# Plot KMeans clusters
plt.figure(figsize=(10, 7))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters_kmeans, cmap='rainbow')
plt.title(f'KMeans Clustering Visualization\nSilhouette Score: {sil_score_kmeans:.3f}\nTF-IDF Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()
plt.savefig(os.path.join(output_dir, 'KMeans_Clustering_TFIDF.png'))
plt.show()

# 2. DBSCAN Clustering
dbscan = DBSCAN(eps=1.5, min_samples=2)
clusters_dbscan = dbscan.fit_predict(X_reduced)

# Evaluate DBSCAN only if there are more than one cluster
if len(np.unique(clusters_dbscan)) > 1:
    sil_score_dbscan, nmi_dbscan, ari_dbscan, fmi_dbscan = evaluate_clustering(pseudo_true_labels, clusters_dbscan,
                                                                               "DBSCAN")
else:
    print("DBSCAN did not find enough clusters to calculate evaluation metrics.")
    sil_score_dbscan = None

# Plot DBSCAN clusters
plt.figure(figsize=(10, 7))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters_dbscan, cmap='rainbow')
if sil_score_dbscan is not None:
    plt.title(f'DBSCAN Clustering Visualization\nSilhouette Score: {sil_score_dbscan:.3f}\nTF-IDF Embeddings')
else:
    plt.title('DBSCAN Clustering Visualization\nTF-IDF Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()
plt.savefig(os.path.join(output_dir, 'DBSCAN_Clustering_TFIDF.png'))
plt.show()

# 3. Hierarchical Agglomerative Clustering (HAC)
hac = AgglomerativeClustering(n_clusters=3)
clusters_hac = hac.fit_predict(X_reduced)

# Evaluate HAC
sil_score_hac, nmi_hac, ari_hac, fmi_hac = evaluate_clustering(pseudo_true_labels, clusters_hac, "HAC")

# Plot HAC clusters
plt.figure(figsize=(10, 7))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters_hac, cmap='rainbow')
plt.title(f'HAC Clustering Visualization\nSilhouette Score: {sil_score_hac:.3f}\nTF-IDF Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()
plt.savefig(os.path.join(output_dir, 'HAC_Clustering_TFIDF.png'))
plt.show()
