import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load Dataset
df = pd.read_csv('earthquake_data.csv')  # Replace with your actual file

# 2. Display Initial Info
print(df.head())
print(df.info())
print(df.describe())

# 3. Clean & Handle Missing Values
df.dropna(subset=['latitude', 'longitude', 'mag', 'depth'], inplace=True)
df['place'].fillna('Unknown', inplace=True)

# 4. Feature Engineering
df['year'] = pd.to_datetime(df['time']).dt.year
df['month'] = pd.to_datetime(df['time']).dt.month
df['day'] = pd.to_datetime(df['time']).dt.day
df['hour'] = pd.to_datetime(df['time']).dt.hour

# 5. Ensuring Consistency
df = df[df['mag'] > 0]  # Removing zero/negative magnitudes
df = df[df['depth'] >= 0]

# 6. Summary Statistics
print("Magnitude Distribution:\n", df['mag'].describe())
print("Depth Distribution:\n", df['depth'].describe())

# 7. Visualize Patterns & Trends
plt.figure(figsize=(12,6))
sns.histplot(df['mag'], bins=30, kde=True)
plt.title('Magnitude Distribution')
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='longitude', y='latitude', hue='mag', size='depth', alpha=0.6, palette='viridis')
plt.title('Earthquake Locations and Magnitudes')
plt.show()

# 8. Outliers & Transformations
df['log_depth'] = np.log1p(df['depth'])  # log transform for skewed depth

# 9. PCA for Dimensionality Reduction (optional)
features = ['mag', 'depth', 'year', 'month', 'hour']
x = StandardScaler().fit_transform(df[features])
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
df['PC1'] = principalComponents[:, 0]
df['PC2'] = principalComponents[:, 1]

plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', hue='mag', data=df, palette='coolwarm')
plt.title('PCA of Earthquake Features')
plt.show()
