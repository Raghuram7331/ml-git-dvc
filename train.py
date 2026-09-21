import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("data/iris.csv")

# Separate features and target
X = df.drop("species", axis=1)
y = df["species"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Random Forest Model Version 2")
print("Number of trees:", model.n_estimators)
print("Maximum depth:", model.max_depth)
print("Accuracy:", accuracy)

# Save the trained model
joblib.dump(model, "models/random_forest.pkl")

print("Model saved successfully to models/random_forest.pkl")
