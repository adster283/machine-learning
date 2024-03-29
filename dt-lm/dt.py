import pandas as pd
import math
import os
import sys


# Importing the dataset
def load_csv(file_name):
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, f'../Files/data_part2/{file_name}')
    print(csv_file_path)
    
    data = pd.read_csv(csv_file_path)

    return data


# Calulates the entropy of a given group
def entropy(group):
    n = len(group)
    if n == 0:
        return 0
    p0 = group.count(0) / n
    p1 = group.count(1) / n
    
    if p0 == 0 or p1 == 0:
        return 0
    
    return -p0 * math.log2(p0) - p1 * math.log2(p1)


# Calculates the information gain of a given split
def information_gain(node, left, right):
    parent = entropy([row[-1] for row in node['data']])
    left_entropy = entropy([row[-1] for row in left])
    right_entropy = entropy([row[-1] for row in right])
    
    total_instance = len(node["data"])
    
    information_gain = parent - ((left_entropy * len(left) / total_instance) + (right_entropy * len(right) / total_instance))
    return information_gain


# Splits a given node into a left and right child node
def split_data(data, index, value):
    left, right = list(), list()
    for row in data:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right
    

# Determines where to split the data for a given node
def get_split(data):
    best_index, best_value, best_score, best_groups = 999, 999, 999, None
    for index in range(len(data[0])-1):
        for row in data:
            groups = split_data(data, index, row[index])
            left_class_values = [row[-1] for row in groups[0]]
            right_class_values = [row[-1] for row in groups[1]]
            entrop = entropy(left_class_values) * len(left_class_values) / len(data) + \
                     entropy(right_class_values) * len(right_class_values) / len(data)
            if entrop < best_score:
                best_index, best_value, best_score, best_groups = index, row[index], entrop, groups
     

    return {'index':best_index, 'value':best_value, 'groups':best_groups, \
        'data':data}   


# Used to create a leaf node an determine the class value that a group of rows will be assigned to
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    group_0 = outcomes.count(0)
    group_1 = outcomes.count(1)
    return {'class': max(set(outcomes), key=outcomes.count), '0': group_0, '1': group_1,}

# 
def split(node, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # Check information gain
    if information_gain(node, left, right) < 0.00001:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # Process left child
    if len(left) <= 1:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], depth+1)
    # Process right child
    if len(right) <= 1:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], depth+1)
        

# Builds the tree by calling the relevant functions
def build_tree(train):
    root = get_split(train)
    split(root, 1)
    return root


# Used to predict the class of a given row a data
def predict(node, row):
    try:
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return predict(node['right'], row)
            else:
                return node['right']
    except:
        return node['class']
    
    
# Prints the tree in an easy to read manner
def print_tree(node, depth=0):
        
    count = 0
    if 'data' in node:
        try:
            test = f"feature {depth+1} ({information_gain(node, node['left']['data'], node['right']['data'])}, Entropy: {entropy([row[-1] for row in node['data']])})"
            if depth > 0:
                print((depth * ' ')*3, f"--{depth}== test --")
            print((depth * ' ')*3, f"feature {depth+1} ({information_gain(node, node['left']['data'], node['right']['data'])}, Entropy: {entropy([row[-1] for row in node['data']])})")
            count += 1
        except:
            pass
        finally:
            if count == 0:
                classification = max(set([row[-1] for row in node['data']]), key=[row[-1] for row in node['data']].count)
                print((depth * ' ')*3, f"--{depth}== {classification} --")
        if node['left']==node['right']:
            print_tree(node['left'], depth+1)
        else:
                print_tree(node['left'], depth+1)
                print_tree(node['right'], depth+1) 
    elif '0' in node:
        print((depth * ' ')*3, "leaf", node['0'], node['1'])
    else:
        print("Something went wrong :()")


# Prints the tree to the desired txt file
def print_tree_to_file(node, file_handle, depth=0):
    count = 0
    if 'data' in node:
        try:
            test = f"feature {depth+1} ({information_gain(node, node['left']['data'], node['right']['data'])}, Entropy: {entropy([row[-1] for row in node['data']])})"
            if depth > 0:
                file_handle.write((depth * ' ')*3 + f"--{depth}== test --\n")
            file_handle.write((depth * ' ')*3 + f"feature {depth+1} ({information_gain(node, node['left']['data'], node['right']['data'])}, Entropy: {entropy([row[-1] for row in node['data']])})\n")
            count += 1
        except:
            pass
        finally:
            if count == 0:
                classification = max(set([row[-1] for row in node['data']]), key=[row[-1] for row in node['data']].count)
                file_handle.write((depth * ' ')*3 + f"--{depth}== {classification} --\n")
        if node['left']==node['right']:
            print_tree_to_file(node['left'], file_handle, depth+1)
        else:
            print_tree_to_file(node['left'], file_handle, depth+1)
            print_tree_to_file(node['right'], file_handle, depth+1)
    elif '0' in node:
        file_handle.write((depth * ' ')*3 + f"leaf {node['0']} {node['1']}\n")
    else:
        file_handle.write("Something went wrong :(\n")

    
def main():
    
    #Handeling the arguments from the terminal
    arguments = sys.argv
    
    train_data = load_csv(arguments[1])
    output_file = arguments[2]
    
    #Creating the trees
    data_set = [tuple(row) for row in train_data.to_numpy()]

    tree = build_tree(data_set)
    print("\n")
    print_tree(tree)
    print("\n")

    #Predict and output the tree to the terminal
    actuals = [row[-1] for row in data_set]
    predicted = []

    for row in data_set:
        predicted.append(predict(tree, row))

    count = 0

    for i in range(len(predicted)):
        if predicted[i] == actuals[i]:
            count += 1

    print(f'Accuracy: {count / len(predicted)}')

    # Calling the function to save the tree to the output file
    with open(output_file, "w") as file:
        print_tree_to_file(tree, file)
        file.write(f'\nAccuracy: {count / len(predicted)}')


if __name__ == '__main__':
    main()