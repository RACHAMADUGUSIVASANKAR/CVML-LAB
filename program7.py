import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt 
df = pd.read_csv("flowers.csv") 
if "Id" in df.columns: 
 df = df.drop(columns=["Id"]) 
X = df.iloc[:, :-1].values 
y = df.iloc[:, -1].values 
X_train, X_test, y_train, y_test = train_test_split( 
 X, y, test_size=0.2, random_state=42, stratify=y) 
dt_model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42) dt_model.fit(X_train, y_train) 
y_pred = dt_model.predict(X_test) 
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred)) print("\nClassification Report:\n", classification_report(y_test, y_pred)) plt.figure(figsize=(12, 8)) 
plot_tree( 
 dt_model, 
 filled=True, 
 feature_names=list(df.columns[:-1]), 
 class_names=sorted(df.iloc[:, -1].unique()) 
) 
plt.show()
