import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Dataset
df = pd.read_csv(r"C:\Users\krish\Downloads\iris.csv")
df = df.drop(columns=["Id"], errors='ignore')

print("Dataset Head:")
print(df.head())

# 2. Feature Matrix (X) and Target (y)
X = df.drop("Species", axis=1)
y = df["Species"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train Model (SVM)
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# 6. Predict
y_pred = model.predict(X_test)

# 7. Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Visualization (optional but good for report)
sns.pairplot(df, hue="Species")
plt.show()