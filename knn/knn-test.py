import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Function to perform min-max normalization
def min_max_normalization(data):
    min_vals = data.min()
    max_vals = data.max()
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data

# Load the CSV files
# Construct file paths using os
script_dir = os.path.dirname(os.path.abspath(__file__))
train_file_path = os.path.join(script_dir, '../Files/data_part1/wine_train.csv')
test_file_path = os.path.join(script_dir, '../Files/data_part1/wine_test.csv')

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Separate features and labels
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']
X_test = test_data.drop(columns=['class'])
y_test = test_data['class']

# Apply min-max normalization to features
X_train_normalized = min_max_normalization(X_train)
X_test_normalized = min_max_normalization(X_test)

# Initialize the KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=13)  # You can adjust the number of neighbors (k) as needed

# Train the classifier
knn_classifier.fit(X_train_normalized, y_train)

# Make predictions
predictions = knn_classifier.predict(X_test_normalized)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Scikit-learn KNN Accuracy: {accuracy}')
