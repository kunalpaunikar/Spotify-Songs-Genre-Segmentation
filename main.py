import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load your dataset
df = pd.read_csv('C:\\Users\\kunal\\Desktop\\majorprojectcorizo\\spotify dataset.csv')

# Display the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Fill missing values for numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Fill missing values for non-numeric columns (optional)
non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
df[non_numeric_columns] = df[non_numeric_columns].fillna('')  # or use df.dropna() to drop rows with missing values

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.select_dtypes(include=[np.number]))

# Create a DataFrame with scaled features
df_scaled = pd.DataFrame(scaled_features, columns=df.select_dtypes(include=[np.number]).columns)

# Distribution of numerical features
df.hist(figsize=(12, 10))
plt.show()

# Plotting correlations between features
plt.figure(figsize=(10, 8))
sns.heatmap(df_scaled.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Apply PCA for dimensionality reduction (optional)
pca = PCA(n_components=2)
pca_features = pca.fit_transform(df_scaled)

# Using the Elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(pca_features)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fit KMeans with optimal clusters (e.g., k=4)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(pca_features)

# Add cluster labels to the dataset
df['Cluster'] = clusters

# Plot clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=clusters, palette='viridis')
plt.title('Clusters of Songs')
plt.show()

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_scaled, df['Cluster'], test_size=0.2, random_state=42)

# Applying KNN for the recommendation model
model = KNeighborsClassifier(n_neighbors=5)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Use the trained model to make recommendations
def recommend_songs(song_features):
    cluster = model.predict(song_features)
    recommendations = df[df['Cluster'] == cluster[0]]
    return recommendations[['track_name', 'track_artist']]  # Customize as per your dataset

# Example usage:
# song_features = np.array([some_features]).reshape(1, -1)
# recommendations = recommend_songs(song_features)
# print(recommendations)