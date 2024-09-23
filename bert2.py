# import os
# import re
# import pickle
# import torch
# import numpy as np
# from bs4 import BeautifulSoup
# from transformers import AutoTokenizer, AutoModel
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
# from sklearn.metrics import silhouette_score, pairwise_distances
#
# # Define the directory where the text files are stored
# data_dir = 'C:/Users/asus/Downloads/clustering/archive (3)'  # Update this to your dataset's directory path
#
# # Get all file names from the directory
# file_names = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
#
# # Load the content from all the files
# documents = []
# for file_name in file_names:
#     with open(os.path.join(data_dir, file_name), 'r', encoding='utf-8', errors='ignore') as file:
#         documents.append(file.read())
#
#
# # Define preprocessing steps using a custom approach instead of NLTK
# def preprocess_text(text):
#     # Lowercasing
#     text = text.lower()
#
#     # Removing HTML tags
#     text = BeautifulSoup(text, "html.parser").get_text()
#
#     # Removing URLs
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
#
#     # Removing punctuation and special characters
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#
#     # Removing numbers
#     text = re.sub(r'\d+', '', text)
#
#     return text
#
#
# # Apply preprocessing to all documents
# documents = [preprocess_text(doc) for doc in documents]
#
# # Use Hugging Face tokenizer directly for tokenization
# model_name = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)
#
#
# # Function to compute BERT embeddings
# def get_bert_embeddings(text, tokenizer, model):
#     # Tokenizing using Hugging Face's tokenizer instead of NLTK
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # Use the [CLS] token embedding
#
#
# # Convert each document to its BERT embedding
# document_embeddings = np.array([get_bert_embeddings(doc, tokenizer, model) for doc in documents])
#
# # Save the BERT embeddings to a file
# output_path = 'C:/Users/asus/Downloads/clustering/bert_embeddings.pkl'  # Change this path if you want to save it elsewhere
# with open(output_path, 'wb') as f:
#     pickle.dump(document_embeddings, f)
#
# print(f"BERT embeddings saved to {output_path}")
#
# # Optionally, perform clustering on the embeddings
# # You can use KMeans, DBSCAN, or AgglomerativeClustering as per your previous approach
# # Here is an example with KMeans:
#
# # Reduce dimensionality using PCA before clustering for better visualization
# pca = PCA(n_components=2)
# X_reduced = pca.fit_transform(document_embeddings)
#
# # Apply KMeans clustering
# kmeans = KMeans(n_clusters=3, random_state=42)
# clusters_kmeans = kmeans.fit_predict(X_reduced)
#
# # Calculate silhouette score
# sil_score_kmeans = silhouette_score(X_reduced, clusters_kmeans)
# print(f"Silhouette Score for KMeans: {sil_score_kmeans}")
#
# # Visualize the clusters using PCA
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(10, 7))
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters_kmeans, cmap='rainbow')
# plt.title(f'KMeans Clustering Visualization\nSilhouette Score: {sil_score_kmeans:.3f}')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.colorbar()
# plt.show()

#
# import os
# import re
# import pickle
# import torch
# import numpy as np
# from bs4 import BeautifulSoup
# from transformers import AutoTokenizer, AutoModel
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
# from sklearn.metrics import silhouette_score
#
# # Define the directory where the text files are stored
# data_dir = 'C:/Users/asus/Downloads/clustering/archive (3)'  # Update this to your dataset's directory path
#
# # Get all file names from the directory
# file_names = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
#
# # Load the content from all the files
# documents = []
# for file_name in file_names:
#     with open(os.path.join(data_dir, file_name), 'r', encoding='utf-8', errors='ignore') as file:
#         documents.append(file.read())
#
#
# # Define preprocessing steps using a custom approach instead of NLTK
# def preprocess_text(text):
#     # Lowercasing
#     text = text.lower()
#
#     # Removing HTML tags
#     text = BeautifulSoup(text, "html.parser").get_text()
#
#     # Removing URLs
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
#
#     # Removing punctuation and special characters
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#
#     # Removing numbers
#     text = re.sub(r'\d+', '', text)
#
#     return text
#
#
# # Apply preprocessing to all documents
# documents = [preprocess_text(doc) for doc in documents]
#
# # Path to save/load BERT embeddings
# output_path = 'C:/Users/asus/Downloads/clustering/bert_embeddings.pkl'
#
# # Check if BERT embeddings already exist
# if os.path.exists(output_path):
#     # Load saved BERT embeddings
#     with open(output_path, 'rb') as f:
#         document_embeddings = pickle.load(f)
#     print(f"Loaded BERT embeddings from {output_path}")
# else:
#     # If not found, generate BERT embeddings
#     print("Generating BERT embeddings...")
#     model_name = "bert-base-uncased"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)
#
#
#     # Function to compute BERT embeddings
#     def get_bert_embeddings(text, tokenizer, model):
#         inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # Use the [CLS] token embedding
#
#
#     # Convert each document to its BERT embedding
#     document_embeddings = np.array([get_bert_embeddings(doc, tokenizer, model) for doc in documents])
#
#     # Save the BERT embeddings to a file
#     with open(output_path, 'wb') as f:
#         pickle.dump(document_embeddings, f)
#
#     print(f"BERT embeddings saved to {output_path}")
#
# # Perform dimensionality reduction using PCA (for visualization and clustering efficiency)
# pca = PCA(n_components=2)
# X_reduced = pca.fit_transform(document_embeddings)
#
# # 1. KMeans Clustering
# kmeans = KMeans(n_clusters=3, random_state=42)
# clusters_kmeans = kmeans.fit_predict(X_reduced)
#
# # Calculate silhouette score for KMeans
# sil_score_kmeans = silhouette_score(X_reduced, clusters_kmeans)
# print(f"Silhouette Score for KMeans: {sil_score_kmeans}")
#
# # 2. DBSCAN Clustering
# dbscan = DBSCAN(eps=1.5, min_samples=2)
# clusters_dbscan = dbscan.fit_predict(X_reduced)
#
# # Calculate silhouette score for DBSCAN if more than one cluster is found
# if len(np.unique(clusters_dbscan)) > 1:
#     sil_score_dbscan = silhouette_score(X_reduced, clusters_dbscan)
#     print(f"Silhouette Score for DBSCAN: {sil_score_dbscan}")
# else:
#     print("DBSCAN did not find enough clusters to calculate Silhouette Score.")
#
# # 3. Hierarchical Agglomerative Clustering (HAC)
# hac = AgglomerativeClustering(n_clusters=3)
# clusters_hac = hac.fit_predict(X_reduced)
#
# # Calculate silhouette score for HAC
# sil_score_hac = silhouette_score(X_reduced, clusters_hac)
# print(f"Silhouette Score for HAC: {sil_score_hac}")
#
# # Visualization
# import matplotlib.pyplot as plt
#
# # Plot KMeans clusters
# plt.figure(figsize=(10, 7))
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters_kmeans, cmap='rainbow')
# plt.title(f'KMeans Clustering Visualization\nSilhouette Score: {sil_score_kmeans:.3f}')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.colorbar()
# plt.show()
#
# # Plot DBSCAN clusters
# plt.figure(figsize=(10, 7))
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters_dbscan, cmap='rainbow')
# plt.title('DBSCAN Clustering Visualization')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.colorbar()
# plt.show()
#
# # Plot HAC clusters
# plt.figure(figsize=(10, 7))
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters_hac, cmap='rainbow')
# plt.title(f'HAC Clustering Visualization\nSilhouette Score: {sil_score_hac:.3f}')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.colorbar()
# plt.show()


# import os
# import re
# import pickle
# import torch
# import numpy as np
# from bs4 import BeautifulSoup
# from transformers import AutoTokenizer, AutoModel
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt
#
# # Define the directory where the text files are stored
# data_dir = 'C:/Users/asus/Downloads/clustering/archive (3)'  # Update this to your dataset's directory path
#
# # Get all file names from the directory
# file_names = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
#
# # Load the content from all the files
# documents = []
# for file_name in file_names:
#     with open(os.path.join(data_dir, file_name), 'r', encoding='utf-8', errors='ignore') as file:
#         documents.append(file.read())
#
#
# # Define preprocessing steps using a custom approach
# def preprocess_text(text):
#     # Lowercasing
#     text = text.lower()
#
#     # Removing HTML tags
#     text = BeautifulSoup(text, "html.parser").get_text()
#
#     # Removing URLs
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
#
#     # Removing punctuation and special characters
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#
#     # Removing numbers
#     text = re.sub(r'\d+', '', text)
#
#     return text
#
#
# # Apply preprocessing to all documents
# documents = [preprocess_text(doc) for doc in documents]
#
# # Path to save/load BERT embeddings
# pkl_path = 'C:/Users/asus/Downloads/clustering/20newsgroup_bert_embeddings.pkl'
# txt_path = 'C:/Users/asus/Downloads/clustering/20newsgroup_bert_embeddings.txt'
#
# # Check if BERT embeddings already exist
# if os.path.exists(pkl_path):
#     # Load saved BERT embeddings
#     with open(pkl_path, 'rb') as f:
#         document_embeddings = pickle.load(f)
#     print(f"Loaded BERT embeddings from {pkl_path}")
# else:
#     # If not found, generate BERT embeddings
#     print("Generating BERT embeddings...")
#     model_name = "bert-base-uncased"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)
#
#
#     # Function to compute BERT embeddings
#     def get_bert_embeddings(text, tokenizer, model):
#         inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # Use the [CLS] token embedding
#
#
#     # Convert each document to its BERT embedding
#     document_embeddings = np.array([get_bert_embeddings(doc, tokenizer, model) for doc in documents])
#
#     # Save the BERT embeddings to a .pkl file
#     with open(pkl_path, 'wb') as f:
#         pickle.dump(document_embeddings, f)
#     print(f"BERT embeddings saved to {pkl_path}")
#
#     # Also save the embeddings to a .txt file for inspection
#     with open(txt_path, 'w') as f:
#         for i, embedding in enumerate(document_embeddings):
#             f.write(f"Document {i + 1} Embedding:\n")
#             np.savetxt(f, embedding, delimiter=',', fmt='%f')
#             f.write("\n")
#     print(f"BERT embeddings also saved as text at {txt_path}")
#
# # Perform dimensionality reduction using PCA (for visualization and clustering efficiency)
# pca = PCA(n_components=2)
# X_reduced = pca.fit_transform(document_embeddings)
#
# # Output directory for images
# output_dir = 'C:/Users/asus/Downloads/clustering'
#
# # 1. KMeans Clustering
# kmeans = KMeans(n_clusters=3, random_state=42)
# clusters_kmeans = kmeans.fit_predict(X_reduced)
#
# # Calculate silhouette score for KMeans
# sil_score_kmeans = silhouette_score(X_reduced, clusters_kmeans)
# print(f"Silhouette Score for KMeans: {sil_score_kmeans}")
#
# # Plot KMeans clusters
# plt.figure(figsize=(10, 7))
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters_kmeans, cmap='rainbow')
# plt.title(f'KMeans Clustering Visualization\nSilhouette Score: {sil_score_kmeans:.3f}\nBERT Embeddings')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.colorbar()
# plt.savefig(os.path.join(output_dir, 'KMeans_Clustering_BERT.png'))
# plt.show()
#
# # 2. DBSCAN Clustering
# dbscan = DBSCAN(eps=1.5, min_samples=2)
# clusters_dbscan = dbscan.fit_predict(X_reduced)
#
# # Calculate silhouette score for DBSCAN if more than one cluster is found
# if len(np.unique(clusters_dbscan)) > 1:
#     sil_score_dbscan = silhouette_score(X_reduced, clusters_dbscan)
#     print(f"Silhouette Score for DBSCAN: {sil_score_dbscan}")
# else:
#     sil_score_dbscan = None
#     print("DBSCAN did not find enough clusters to calculate Silhouette Score.")
#
# # Plot DBSCAN clusters
# plt.figure(figsize=(10, 7))
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters_dbscan, cmap='rainbow')
# if sil_score_dbscan is not None:
#     plt.title(f'DBSCAN Clustering Visualization\nSilhouette Score: {sil_score_dbscan:.3f}\nBERT Embeddings')
# else:
#     plt.title('DBSCAN Clustering Visualization\nBERT Embeddings')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.colorbar()
# plt.savefig(os.path.join(output_dir, 'DBSCAN_Clustering_BERT.png'))
# plt.show()
#
# # 3. Hierarchical Agglomerative Clustering (HAC)
# hac = AgglomerativeClustering(n_clusters=3)
# clusters_hac = hac.fit_predict(X_reduced)
#
# # Calculate silhouette score for HAC
# sil_score_hac = silhouette_score(X_reduced, clusters_hac)
# print(f"Silhouette Score for HAC: {sil_score_hac}")
#
# # Plot HAC clusters
# plt.figure(figsize=(10, 7))
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters_hac, cmap='rainbow')
# plt.title(f'HAC Clustering Visualization\nSilhouette Score: {sil_score_hac:.3f}\nBERT Embeddings')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.colorbar()
# plt.savefig(os.path.join(output_dir, 'HAC_Clustering_BERT.png'))
# plt.show()


import os
import re
import pickle
import torch
import numpy as np
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score
import matplotlib.pyplot as plt

# Define the directory where the text files are stored
data_dir = 'C:/Users/asus/Downloads/clustering/archive (3)'  # Update this to your dataset's directory path

# Get all file names from the directory
file_names = [f for f in os.listdir(data_dir) if f.endswith('.txt')]

# Load the content from all the files
documents = []
for file_name in file_names:
    with open(os.path.join(data_dir, file_name), 'r', encoding='utf-8', errors='ignore') as file:
        documents.append(file.read())


# Define preprocessing steps using a custom approach
def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    # Removing HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Removing URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Removing punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Removing numbers
    text = re.sub(r'\d+', '', text)

    return text


# Apply preprocessing to all documents
documents = [preprocess_text(doc) for doc in documents]

# Path to save/load BERT embeddings
pkl_path = 'C:/Users/asus/Downloads/clustering/20newsgroup_bert_embeddings.pkl'
txt_path = 'C:/Users/asus/Downloads/clustering/20newsgroup_bert_embeddings.txt'

# Check if BERT embeddings already exist
if os.path.exists(pkl_path):
    # Load saved BERT embeddings
    with open(pkl_path, 'rb') as f:
        document_embeddings = pickle.load(f)
    print(f"Loaded BERT embeddings from {pkl_path}")
else:
    # If not found, generate BERT embeddings
    print("Generating BERT embeddings...")
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)


    # Function to compute BERT embeddings
    def get_bert_embeddings(text, tokenizer, model):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # Use the [CLS] token embedding


    # Convert each document to its BERT embedding
    document_embeddings = np.array([get_bert_embeddings(doc, tokenizer, model) for doc in documents])

    # Save the BERT embeddings to a .pkl file
    with open(pkl_path, 'wb') as f:
        pickle.dump(document_embeddings, f)
    print(f"BERT embeddings saved to {pkl_path}")

    # Also save the embeddings to a .txt file for inspection
    with open(txt_path, 'w') as f:
        for i, embedding in enumerate(document_embeddings):
            f.write(f"Document {i + 1} Embedding:\n")
            np.savetxt(f, embedding, delimiter=',', fmt='%f')
            f.write("\n")
    print(f"BERT embeddings also saved as text at {txt_path}")

# Perform dimensionality reduction using PCA (for visualization and clustering efficiency)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(document_embeddings)

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
plt.title(f'KMeans Clustering Visualization\nSilhouette Score: {sil_score_kmeans:.3f}\nBERT Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()
plt.savefig(os.path.join(output_dir, 'KMeans_Clustering_BERT.png'))
plt.show()

# 2. DBSCAN Clustering
dbscan = DBSCAN(eps=1.5, min_samples=2)
clusters_dbscan = dbscan.fit_predict(X_reduced)

# Evaluate DBSCAN only if there are more than one cluster
if len(np.unique(clusters_dbscan)) > 1:
    sil_score_dbscan, nmi_dbscan, ari_dbscan, fmi_dbscan = evaluate_clustering(pseudo_true_labels, clusters_dbscan,
                                                                               "DBSCAN")
else:
    sil_score_dbscan = None
    print("DBSCAN did not find enough clusters to calculate evaluation metrics.")

# Plot DBSCAN clusters
plt.figure(figsize=(10, 7))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters_dbscan, cmap='rainbow')
if sil_score_dbscan is not None:
    plt.title(f'DBSCAN Clustering Visualization\nSilhouette Score: {sil_score_dbscan:.3f}\nBERT Embeddings')
else:
    plt.title('DBSCAN Clustering Visualization\nBERT Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()
plt.savefig(os.path.join(output_dir, 'DBSCAN_Clustering_BERT.png'))
plt.show()

# 3. Hierarchical Agglomerative Clustering (HAC)
hac = AgglomerativeClustering(n_clusters=3)
clusters_hac = hac.fit_predict(X_reduced)

# Evaluate HAC
sil_score_hac, nmi_hac, ari_hac, fmi_hac = evaluate_clustering(pseudo_true_labels, clusters_hac, "HAC")

# Plot HAC clusters
plt.figure(figsize=(10, 7))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters_hac, cmap='rainbow')
plt.title(f'HAC Clustering Visualization\nSilhouette Score: {sil_score_hac:.3f}\nBERT Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()
plt.savefig(os.path.join(output_dir, 'HAC_Clustering_BERT.png'))
plt.show()
