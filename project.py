import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


import os
for dirname, _, filenames in os.walk('/data/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/data/input/marmara-region-earthquakes-apr-2324-2025/deprem_son24saat_duzenli.csv')
df.head()
df.tail()
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
df.dtypes
df.shape
df.columns
plt.figure(figsize=(10, 6))
df['Olus_Zamani'] = pd.to_datetime(df['Olus_Zamani'])  
df['Year'] = df['Olus_Zamani'].dt.year
plt.title('Number of Earthquakes per Year')
sns.countplot(data=df, x='Year', palette="Blues")
plt.xlabel('Year')
plt.ylabel('Count of Earthquakes')
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Boylam', y='Enlem', hue='Buyukluk', size='Derinlik_km', sizes=(20, 200), palette="viridis")
plt.title('Earthquake Locations (Latitude vs Longitude)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
plt.figure(figsize=(10, 6))
sns.histplot(df['Derinlik_km'], kde=True, color="purple", bins=30)
plt.title('Earthquake Depth Distribution')
plt.xlabel('Depth (km)')
plt.ylabel('Frequency')
plt.show()
plt.figure(figsize=(10, 6))
sns.histplot(df['Buyukluk'], kde=True, color="green", bins=30)
plt.title('Earthquake Magnitude Distribution')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.show()
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Yer', palette="coolwarm")
plt.title('Earthquake Locations/Regions')
plt.xlabel('Location/Region')
plt.ylabel('Count of Earthquakes')
plt.xticks(rotation=45)
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print(df.columns)
df["Olus_Zamani"] = pd.to_datetime(df["Olus_Zamani"])
df["Saat"] = df["Olus_Zamani"].dt.hour
df["Gun"] = df["Olus_Zamani"].dt.day
df["Dakika"] = df["Olus_Zamani"].dt.minute
df["Haftanin_Gunu"] = df["Olus_Zamani"].dt.day_name()
def classify_magnitude(mag):
    if mag < 3.5:
        return "Small"
    elif 3.5 <= mag < 5.0:
        return "Moderate"
    else:
        return "Large"

df["Buyukluk_Sinifi"] = df["Buyukluk"].apply(classify_magnitude)
def classify_depth(depth):
    if depth <= 15:
        return "Shallow"
    elif 15 < depth <= 50:
        return "Intermediate"
    else:
        return "Deep"

df["Derinlik_Sinifi"] = df["Derinlik_km"].apply(classify_depth)
def region_label(lat, lon):
    if lat > 39 and lon > 30:
        return "Central_North"
    elif lat < 37:
        return "South"
    elif lon < 27:
        return "West"
    else:
        return "Other"

df["Bolge"] = df.apply(lambda row: region_label(row["Enlem"], row["Boylam"]), axis=1)
print(df.head())
from sklearn.preprocessing import StandardScaler
X = df[["Derinlik_km", "Saat", "Gun", "Dakika", "Haftanin_Gunu", "Buyukluk_Sinifi", "Derinlik_Sinifi", "Bolge"]]
X = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled[:5])
from sklearn.model_selection import train_test_split
y = df["Buyukluk_Sinifi"]
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=537)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=537)
models = {
    "Logistic Regression": LogisticRegression(
        penalty='l2',           
        C=1.0,                  
        max_iter=300,           
        solver='lbfgs',         
        multi_class='multinomial'  
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,       
        max_depth=12,           
        min_samples_split=4,    
        min_samples_leaf=2,     
        bootstrap=True,         
        random_state=42         
    ),

    "Support Vector Machine": SVC(
        kernel='rbf',           
        C=1.0,                 
        gamma='scale',          
        probability=True       
    )
}
accuracies = {}

for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    
    print(f"Accuracy: {acc:.4f}")  # Print accuracy
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))  # Precision, Recall, F1-score
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=["Small", "Moderate", "Large"]))
accuracies_percent = {model: acc * 100 for model, acc in accuracies.items()}
plt.figure(figsize=(8, 5))
bars = plt.barh(list(accuracies_percent.keys()), accuracies_percent.values(), color='skyblue')
plt.xlabel('Accuracy (%)')
plt.title('Model Accuracy Comparison')
for bar in bars:
    plt.text(bar.get_width() - 3, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}%', 
             va='center', ha='right', color='black', fontweight='bold')
plt.show()




