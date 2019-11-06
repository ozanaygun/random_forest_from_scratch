# random_forest_from_scratch
This repo contains my implmentation of [Random Forest](https://en.wikipedia.org/wiki/Random_forest) in python as a package. It's not completely from scratch as I've used numpy and pandas for data handling.

## Install
1. Download or Clone the repo.
2. Copy the [random_forest](https://github.com/geekyJock8/random_forest_from_scratch/tree/master/random_forest) folder into the root of your project.

## Use
### Import
from random_forest import RandomForest
### Class RandomForest
RandomForest(n_estimators, sample_size, min_leaf)
#### Parametes
1. **n_estimators** : This is basically the count of the trees in our forest.
2. **sample_size** : This specifies the batch size of data for a tree. Data is sampled with replacment from the training dataset provided.
3. **min_leaf** : The size of smallest node in the tree.
#### Methods
1. **fit(x, y)** : Build a forest of trees from the training set (x, y). x and y are of the type dataframe.
2. **predict(x)** : Return a numpy.array() of predictions. x is of the type dataframe.

## TODO
1. Add more criterion.
2. Implement Classifier.
3. Add support for non numeric values.
4. Handling of other types of data inputs.

#
*Contributions are welcome*