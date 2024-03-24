import os
import sys
import pandas as pd
import math

sorted_distances = []

# Allows for only giving the name of the file and not having to give the full path
def load_csv(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, f'../Files/data_part1/{file_name}')
    data = pd.read_csv(csv_file_path)

    # Exclude the last column (class labels) from normalization
    normalized_data = min_max_normalization(data.iloc[:, :-1])

    # Add the class labels back to the normalized data
    normalized_data['class'] = data['class']

    return normalized_data


# Normalises the data so we can get better results of the data
def min_max_normalization(data):
    data_copy = data.copy()
    min_vals = data_copy.min()
    max_vals = data_copy.max()
    normalized_data = (data_copy - min_vals) / (max_vals - min_vals)

    # Explicitly cast the normalized data to ensure compatibility
    normalized_data = normalized_data.astype(float)

    return normalized_data


def create_points_list(data, class_label):
    class_data = data[data['class'] == class_label]
    class_data = class_data.drop(columns=['class'])
    points_list = [tuple(row) for row in class_data.to_numpy()]
    return points_list


# Classifies a point as being inside one of the classes
def classifyAPoint(points, p, k=3):
    distances = []
    for group_label, group_points in points.items():
        for feature in group_points:
            euclidean_distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(feature, p)))
            distances.append((euclidean_distance, group_label))

    distances = sorted(distances)[:k]
    sorted_distances.append(distances)

    # Count the frequency of each group in the k nearest neighbors
    frequencies = {}
    for d in distances:
        if d[1] in frequencies:
            frequencies[d[1]] += 1
        else:
            frequencies[d[1]] = 1

    # Return the group with the highest frequency
    return max(frequencies, key=frequencies.get)


def main():
    arguments = sys.argv

    # Loading in the files
    train_file = load_csv(arguments[1])
    test_file = load_csv(arguments[2])
    output_file = arguments[3]

    # Creating the lists of training points for each class
    class1 = create_points_list(train_file, 1)
    class2 = create_points_list(train_file, 2)
    class3 = create_points_list(train_file, 3)

    # Dictionary of training points for each class
    points = {1: class1, 2: class2, 3: class3}

    # Creating a list of test points
    test_points = [tuple(row) for row in test_file.to_numpy()]

    # Setting up the k value
    k = int(arguments[4])

    # Classifying test points
    list_of_classifications = []

    for test_point in test_points:
        list_of_classifications.append(classifyAPoint(points, test_point, k))


    # Calculating the accuracy
    actuals = test_file['class'].to_list()
    count = 0

    for i in range(len(list_of_classifications)):
        if list_of_classifications[i] == actuals[i]:
            count += 1

    print(f'Accuracy: {count / len(list_of_classifications)}')

    # Creating the output file
    # Create headings for distances
    distance_headings = ['Distance' + str(i + 1) for i in range(k)]

    # Export results to output file
    my_df = {'Y': pd.Series(test_file['class']),
             'Predictions': pd.Series(list_of_classifications)}
    
    # Add distance columns
    for i in range(k):
        my_df[distance_headings[i]] = pd.Series([distances[i] if len(distances) > i else np.nan for distances in sorted_distances])

    df = pd.DataFrame(my_df)
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()
