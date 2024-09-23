from transformers import GPT2Tokenizer, GPT2Model
import os
import re
import pickle
import torch
import numpy as np
from bs4 import BeautifulSoup
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score
import matplotlib.pyplot as plt

# Load GPT-2 model
model_name = "gpt2"  # Replace with any other model on Hugging Face, if desired
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)

# Add padding token if it doesn't exist
tokenizer.pad_token = tokenizer.eos_token

# Define data directory
data_dir = 'C:/Users/asus/Downloads/clustering/archive (3)'

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

# Get GPT-2 embeddings
def get_gpt_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

document_embeddings = np.array([get_gpt_embeddings(doc, tokenizer, model) for doc in documents])

# Save embeddings
output_path = 'C:/Users/asus/Downloads/clustering/20newsgroup_gpt2_embeddings.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(document_embeddings, f)

# Dimensionality reduction
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(document_embeddings)

# Output directory for saving images
output_dir = 'C:/Users/asus/Downloads/clustering'

# Function for evaluating clustering methods
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

# Since we don't have true labels, we use the KMeans assignments as pseudo ground truth
pseudo_true_labels = KMeans(n_clusters=3, random_state=42).fit_predict(X_reduced)

# 1. KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters_kmeans = kmeans.fit_predict(X_reduced)

# Evaluate KMeans
sil_score_kmeans, nmi_kmeans, ari_kmeans, fmi_kmeans = evaluate_clustering(pseudo_true_labels, clusters_kmeans, "KMeans")

# Plot KMeans clusters
plt.figure(figsize=(10, 7))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters_kmeans, cmap='rainbow')
plt.title(f'KMeans Clustering Visualization\nSilhouette Score: {sil_score_kmeans:.3f}\nGPT-2 Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()
plt.savefig(os.path.join(output_dir, 'KMeans_Clustering_GPT2.png'))
plt.show()

# 2. DBSCAN Clustering
dbscan = DBSCAN(eps=1.5, min_samples=2)
clusters_dbscan = dbscan.fit_predict(X_reduced)

# Evaluate DBSCAN only if there are more than one cluster
if len(np.unique(clusters_dbscan)) > 1:
    sil_score_dbscan, nmi_dbscan, ari_dbscan, fmi_dbscan = evaluate_clustering(pseudo_true_labels, clusters_dbscan, "DBSCAN")
else:
    sil_score_dbscan = None
    print("DBSCAN did not find enough clusters to calculate evaluation metrics.")

# Plot DBSCAN clusters
plt.figure(figsize=(10, 7))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters_dbscan, cmap='rainbow')
if sil_score_dbscan is not None:
    plt.title(f'DBSCAN Clustering Visualization\nSilhouette Score: {sil_score_dbscan:.3f}\nGPT-2 Embeddings')
else:
    plt.title('DBSCAN Clustering Visualization\nGPT-2 Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()
plt.savefig(os.path.join(output_dir, 'DBSCAN_Clustering_GPT2.png'))
plt.show()

# 3. Hierarchical Agglomerative Clustering (HAC)
hac = AgglomerativeClustering(n_clusters=3)
clusters_hac = hac.fit_predict(X_reduced)

# Evaluate HAC
sil_score_hac, nmi_hac, ari_hac, fmi_hac = evaluate_clustering(pseudo_true_labels, clusters_hac, "HAC")

# Plot HAC clusters
plt.figure(figsize=(10, 7))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters_hac, cmap='rainbow')
plt.title(f'HAC Clustering Visualization\nSilhouette Score: {sil_score_hac:.3f}\nGPT-2 Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()
plt.savefig(os.path.join(output_dir, 'HAC_Clustering_GPT2.png'))
plt.show()
