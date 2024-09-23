import os
import re
import pickle
import numpy as np
from bs4 import BeautifulSoup
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score
import matplotlib.pyplot as plt

# Define the directory where the text files are stored
data_dir = 'C:/Users/asus/Downloads/clustering/archive (3)'  # Update this to your dataset's directory path

# Load GloVe pre-trained embeddings
glove_path = 'C:/Users/asus/Downloads/glove.6B/glove.6B.300d.txt'  # Update this path to your GloVe location
embedding_dim = 300

# Load GloVe embeddings into a dictionary
glove_embeddings = {}
with open(glove_path, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        glove_embeddings[word] = vector

# Load and preprocess the text files
file_names = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
documents = []
for file_name in file_names:
    with open(os.path.join(data_dir, file_name), 'r', encoding='utf-8', errors='ignore') as file:
        text = file.read().lower()
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\d+', '', text)
        documents.append(text)


# Compute document embeddings by averaging GloVe embeddings of the words
def get_glove_embedding(doc, glove_embeddings, embedding_dim):
    words = doc.split()
    valid_embeddings = [glove_embeddings[word] for word in words if word in glove_embeddings]
    return np.mean(valid_embeddings, axis=0) if valid_embeddings else np.zeros(embedding_dim)


document_embeddings = np.array([get_glove_embedding(doc, glove_embeddings, embedding_dim) for doc in documents])

# Save embeddings
output_path = 'C:/Users/asus/Downloads/clustering/20newsgroup_glove_embeddings.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(document_embeddings, f)

# Perform dimensionality reduction
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(document_embeddings)


# Function for evaluating clustering methods
def evaluate_clustering(true_labels, predicted_labels, method_name):
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


# Clustering using KMeans, DBSCAN, and HAC
pseudo_true_labels = KMeans(n_clusters=3, random_state=42).fit_predict(X_reduced)

# 1. KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters_kmeans = kmeans.fit_predict(X_reduced)
sil_score_kmeans, nmi_kmeans, ari_kmeans, fmi_kmeans = evaluate_clustering(pseudo_true_labels, clusters_kmeans, "KMeans")

# Plot KMeans
plt.figure(figsize=(10, 7))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters_kmeans, cmap='rainbow')
plt.title(f'KMeans Clustering Visualization\nSilhouette Score: {sil_score_kmeans:.3f}\nGloVe Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()
plt.savefig('C:/Users/asus/Downloads/clustering/KMeans_Clustering_GloVe.png')
plt.show()

# 2. DBSCAN Clustering
dbscan = DBSCAN(eps=1.5, min_samples=2)
clusters_dbscan = dbscan.fit_predict(X_reduced)

if len(np.unique(clusters_dbscan)) > 1:
    sil_score_dbscan, nmi_dbscan, ari_dbscan, fmi_dbscan = evaluate_clustering(pseudo_true_labels, clusters_dbscan, "DBSCAN")
else:
    sil_score_dbscan = None
    print("DBSCAN did not find enough clusters to calculate evaluation metrics.")

# Plot DBSCAN clusters
plt.figure(figsize=(10, 7))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters_dbscan, cmap='rainbow')
plt.title(
    f'DBSCAN Clustering Visualization\nSilhouette Score: {sil_score_dbscan if sil_score_dbscan else "N/A"}\nGloVe Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()
plt.savefig('C:/Users/asus/Downloads/clustering/DBSCAN_Clustering_GloVe.png')
plt.show()

# 3. Hierarchical Agglomerative Clustering (HAC)
hac = AgglomerativeClustering(n_clusters=3)
clusters_hac = hac.fit_predict(X_reduced)

sil_score_hac, nmi_hac, ari_hac, fmi_hac = evaluate_clustering(pseudo_true_labels, clusters_hac, "HAC")

# Plot HAC clusters
plt.figure(figsize=(10, 7))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters_hac, cmap='rainbow')
plt.title(f'HAC Clustering Visualization\nSilhouette Score: {sil_score_hac:.3f}\nGloVe Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()
plt.savefig('C:/Users/asus/Downloads/clustering/HAC_Clustering_GloVe.png')
plt.show()
