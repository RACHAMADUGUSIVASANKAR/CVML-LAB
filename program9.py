import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("csv files/pca.csv")
print("Dataset shape:", df.shape)
print(df.head())

df = df.drop(columns=["Formatted Date", "Summary", "Daily Summary"], errors="ignore")

if "Precipitation (mm)" in df.columns:
    df["Rain"] = (df["Precipitation (mm)"] > 0).astype(int)
elif "Precip Type" in df.columns:
    df["Rain"] = df["Precip Type"].fillna("none").apply(lambda x: 1 if x.lower() == "rain" else 0)
else:
    raise ValueError("No precipitation column found to create target!")

df = df.dropna()
X = df.drop("Rain", axis=1)
y = df["Rain"]

for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=60)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization - Weather Data (Rain Classification)")
plt.colorbar(label="Rain (0=No, 1=Yes)")
plt.show()
