import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- CONFIGURATION FOR NICE TABLES ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'left')

# 1. Load the dataset
try:
    df = pd.read_csv("data/titanic_dataset.csv")
    print("Dataset Loaded. Total Passengers:", len(df))
except FileNotFoundError:
    print("Error: Dataset not found in 'data/' folder.")
    exit()

# 2. Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# 3. Drop Cabin column
df.drop("Cabin", axis=1, inplace=True)

# 4. Feature Engineering
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# 5. Convert categorical to numerical
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# 6. Correlation Heatmap
print("\nGenerating Correlation Heatmap...")
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.savefig("visuals/correlation_heatmap.png")
plt.close()

# 7. Select features
passenger_ids = df["PassengerId"]
passenger_names = df["Name"]

X = df.drop(["PassengerId", "Ticket", "Name", "Survived"], axis=1)
y = df["Survived"]

# 8. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 9. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 10. Train Logistic Regression model
print("Training Model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 11. Model Evaluation
y_pred_test = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred_test)
print(f"\nModel Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)
conf_table = pd.DataFrame(
    conf_matrix,
    index=["Actual: No", "Actual: Yes"],
    columns=["Predicted: No", "Predicted: Yes"]
)

print("\n--- CONFUSION MATRIX ---")
print(conf_table)
print("\n" + "=" * 50)

# 12. Feature Importance
feature_imp = pd.DataFrame({
    "Feature": X.columns,
    "Score": model.coef_[0]
}).sort_values(by="Score", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(
    x="Score",
    y="Feature",
    data=feature_imp,
    hue="Feature",
    legend=False,
    palette="viridis"
)
plt.title("Feature Importance")
plt.savefig("visuals/feature_importance.png")
plt.close()

# 13. Predict for all passengers
X_full_scaled = scaler.transform(X)
all_predictions = model.predict(X_full_scaled)

output = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Name": passenger_names,
    "Actual_Survived": y.map({0: "No", 1: "Yes"}),
    "Predicted_Survived": pd.Series(all_predictions).map({0: "No", 1: "Yes"})
})

output = output.sort_values(by="PassengerId")

# Save results
try:
    output.to_csv("outputs/all_passengers_predictions.csv", index=False)
    print("\n--- FINAL PREDICTIONS (First 10 Rows) ---")
    print(output.head(10).to_string(index=False))
    print("\nSuccess! File saved in outputs folder.")
except PermissionError:
    print("\n[ERROR] Close the CSV file and try again.")