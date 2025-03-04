import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("diabetes2.csv", sep= ",")
df.head()
print("Normalize işlemi oncesi veri seti; \n")
df.info()
df.T.describe()
print(df.head()) 
from sklearn.model_selection import train_test_split

X = df.drop(columns=["Outcome"]) 
y = df["Outcome"]#depended variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
df_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
print("Normalize işlemi sonrası veri seti; \n")
df_scaled.info()



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print(report)

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
print("confisuan matrix: \n",cm)

#matrixi daha güzel ve detaylı gösterelim.
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
