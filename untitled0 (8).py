from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Determining the number of clusters, in this case there are 4 (NF, NT, SJ, SP)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X)

# Adding clusters to DataFrame
clustered_data = pd.DataFrame(X, columns=[f'adjective{i+1}' for i in range(8)])
clustered_data['Cluster'] = clusters
clustered_data['Career Inclination'] = career_inclination

# Adjust the PCA to 2 components for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
clustered_data['PC1'] = X_pca[:, 0]
clustered_data['PC2'] = X_pca[:, 1]

# Defining cluster names
cluster_names = {
    0: 'Diplomats (NF)',   # The least ambitious
    1: 'Intellectuals (NT)', # The most ambitious
    2: 'Defenders (SJ)',    # More ambitious than SP
    3: 'Seekers (SP)'      # Less ambitious than SJ
}
clustered_data['Cluster Name'] = clustered_data['Cluster'].map(cluster_names)

custom_palette = {
    'Intellectuals (NT)': 'purple',
    'Seekers (SP)': 'yellow',
    'Diplomats (NF)': 'green',
    'Defenders (SJ)': 'blue'
}

# We assign career ambitions according to the priority of clusters
def assign_career_inclination(cluster_name):
    if cluster_name == 'Intellectuals (NT)':
        return 2  # High level of career ambition
    elif cluster_name == 'Defenders (SJ)':
        return 1  # Average level of career ambition
    elif cluster_name == 'Seekers (SP)':
        return 1  # Medium/Low level
    else:
        return 0  # Low level for NF (Diplomats)

#Setting the level of career ambitions depending on the cluster
clustered_data['Career Inclination Assigned'] = clustered_data['Cluster Name'].apply(assign_career_inclination)

# Visualization of the distribution by career ambitions
plt.figure(figsize=(15, 8))
sns.scatterplot(
    x='PC1', y='PC2',
    hue='Cluster Name',
    style='Career Inclination Assigned',
    data=clustered_data,
    palette=custom_palette,
    s=200)
plt.title('Distribution of clusters by career ambitions')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Clusters and ambitions')
plt.show()
plt.savefig('career_ambitions_clusters.png')
# Show the plot
plt.show()
# Output of the average level of career ambitions for each cluster
for cluster_id, group in clustered_data.groupby('Cluster Name'):
    mean_ambition = group['Career Inclination Assigned'].mean()
    print(f"The average level of career ambitions for {cluster_id}: {mean_ambition:.2f}")
    
