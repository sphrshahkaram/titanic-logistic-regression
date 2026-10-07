import numpy as np
import pandas as pd


# -----------------------------
# 1. Load the data
# -----------------------------

df = pd.read_csv(r"E:\New folder\train.csv")


# -----------------------------
# 2. Clean the data
# -----------------------------

# Cabin has too many missing values, so we remove it.
df = df.drop("Cabin", axis=1)

# Fill missing Age values with the median age.
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with the most common value.
df["Embarked"] = df["Embarked"].fillna("S")

# Convert Sex into numbers.
# male = 0
# female = 1
df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})


# -----------------------------
# 3. Choose features and target
# -----------------------------

features = [
    "Age",
    "Sex",
    "Pclass",
    "Fare",
    "SibSp",
    "Parch"
]

X = df[features].values
y = df["Survived"].values


# -----------------------------
# 4. Split the data
# -----------------------------

np.random.seed(42)

indices = np.arange(len(X))
np.random.shuffle(indices)

train_end = int(0.70 * len(X))
validation_end = int(0.90 * len(X))

train_indices = indices[:train_end]
validation_indices = indices[train_end:validation_end]
test_indices = indices[validation_end:]

X_train = X[train_indices]
y_train = y[train_indices]

X_validation = X[validation_indices]
y_validation = y[validation_indices]

X_test = X[test_indices]
y_test = y[test_indices]


print("Training samples:", len(X_train))
print("Validation samples:", len(X_validation))
print("Test samples:", len(X_test))


# -----------------------------
# 5. Standardize the features
# -----------------------------

# IMPORTANT:
# We calculate mean and std ONLY from training data.

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_validation = (X_validation - mean) / std
X_test = (X_test - mean) / std


# -----------------------------
# 6. Sigmoid function
# -----------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# -----------------------------
# 7. Initialize the model
# -----------------------------

n_features = X_train.shape[1]

w = np.zeros(n_features)
b = 0

learning_rate = 0.01
epochs = 5000


# -----------------------------
# 8. Train the model
# -----------------------------

for epoch in range(epochs):

    # Calculate the score
    z = X_train @ w + b

    # Convert score into probability
    probabilities = sigmoid(z)

    # Calculate the error
    error = probabilities - y_train

    # Calculate gradients
    dw = (X_train.T @ error) / len(y_train)
    db = np.mean(error)

    # Update weights and bias
    w = w - learning_rate * dw
    b = b - learning_rate * db


# -----------------------------
# 9. Make predictions
# -----------------------------

train_probabilities = sigmoid(X_train @ w + b)
validation_probabilities = sigmoid(X_validation @ w + b)
test_probabilities = sigmoid(X_test @ w + b)

train_predictions = (train_probabilities >= 0.5).astype(int)
validation_predictions = (validation_probabilities >= 0.5).astype(int)
test_predictions = (test_probabilities >= 0.5).astype(int)


# -----------------------------
# 10. Calculate accuracy
# -----------------------------

train_accuracy = np.mean(train_predictions == y_train)
validation_accuracy = np.mean(
    validation_predictions == y_validation
)
test_accuracy = np.mean(test_predictions == y_test)


print("\nResults")
print("--------------------")
print("Training accuracy:", train_accuracy)
print("Validation accuracy:", validation_accuracy)
print("Test accuracy:", test_accuracy)


# -----------------------------
# 11. Look at what the model learned
# -----------------------------

print("\nModel parameters")
print("--------------------")
print("Weights:", w)
print("Bias:", b)
