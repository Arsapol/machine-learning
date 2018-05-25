# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
# from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, accuracy_score

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
# from IPython.# display import # display, Image # Allows the use of # display() for DataFrames


# Import supplementary visualization code visuals.py
# import visuals as # vs

# Pretty # display for notebooks
# %matplotlib inline

# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - # display the first record
# display(data.head(n=1))

# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
# vs.distribution(features_log_transformed, transformed = True)

# Visualize skewed continuous features of original data
# # vs.distribution(data)

# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
# # display(features_log_minmax_transform.head(n = 5))

# TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
features_final = pd.get_dummies(features_log_minmax_transform)

# TODO: Encode the 'income_raw' data to numerical values
income = [(lambda price: price=='>50K')(price) for price in income_raw]

# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# Uncomment the following line to see the encoded feature names
#print encoded

# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))



# TODO: Initialize the classifier
clf = RandomForestClassifier()

# TODO: Create the parameters list you wish to tune, using a dictionary if needed.
# HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}
parameters = {'n_estimators': [200, 500],
              'max_depth' : [20,40,60,100,None],
              'random_state' : [46,80],
              'max_features': ['auto', 'sqrt', 'log2'],
              'max_leaf_nodes' : [None, 16, 32, 64],
              'min_samples_split' : [2, 4, 8, 16],
              'min_samples_leaf' : [1, 2, 8 , 32, 64],
              'criterion' : ['entropy','gini'],
              'n_jobs' : [-1]
             }

# TODO: Make an fbeta_score scoring object using make_scorer()
def fbeta_score_function(y_true=None, y_pred=None):
    return fbeta_score(y_true, y_pred, 0.5)
scorer = make_scorer(fbeta_score_function)

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
#grid_obj = GridSearchCV(estimator=clf, param_grid=parameters, scoring=scorer)
n_iter_search = 20
grid_obj = RandomizedSearchCV(clf, param_distributions=parameters, n_iter=n_iter_search)

# TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_
print(best_clf)

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
# print("Unoptimized model\n------")
# print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions.round())))
# print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions.round(), beta = 0.5)))
# print("\nOptimized Model\n------")
# print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions.round())))
# print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions.round(), beta = 0.5)))

print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))