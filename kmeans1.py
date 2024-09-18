import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Create synthetic customer purchase data
np.random.seed(42)  # For reproducibility

# Generate synthetic purchase data
data = {
    'CustomerID': range(1, 101),
    'TotalPurchases': np.random.randint(1, 100, 100),
    'AveragePurchaseValue': np.random.uniform(10.0, 500.0, 100)
}

df = pd.DataFrame(data)

# Step 2: Preprocess the data
# We will scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['TotalPurchases', 'AveragePurchaseValue']])

# Step 3: Determine the optimal number of clusters using the Elbow Method
inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plotting the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.xticks(K)
plt.grid()
plt.show()

# Step 4: Fit K-means with the optimal number of clusters (let's say 3 from the elbow method)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Step 5: Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['TotalPurchases'], df['AveragePurchaseValue'], c=df['Cluster'], cmap='viridis')
plt.title('Customer Clusters')
plt.xlabel('Total Purchases')
plt.ylabel('Average Purchase Value')
plt.colorbar(label='Cluster')
plt.show()

# Step 6: Analyze the clusters
print("Cluster Centers:")
print(scaler.inverse_transform(kmeans.cluster_centers_))

# Show the first few records with their cluster assignment
print(df.head())
