# Before runnign the scripts

You will need to ensure you have installed all dependencies.
```
python3 -m venv myenv
source myenv/bin/activate
```
```
pip install -r requirements.txt
```

Scripts are setup to grab data files out of the Files dir. For instance knn grabs it's files from data_part1 and dt grabs  it's files out of data_part2.

# Running the KNN algorithm

Make sure you are in the knn directory before running the knn.py script.

The script uses the following positional arguments:
    python3 knn.py train.csv test.csv k

```
python3 knn.py wine_train.csv wine_test.csv 3
```

# Running the decision tree algorithm

Make sure you are in the dt-lm directory before running the dt.py script.

The script uses the following positional arguments:
    python3 dt.py train.csv output.txt

An example of a command to run the algorithm is:
```
python3 dt.py rtg_A.csv output.txt
```

# Special thanks to:
I used the code from machinelearningmastery as a base for my decision tree and then modified the code to work with entropy and print out the tree in a human readable way.

https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/