import pandas as pd
import numpy as np
from sklearn.model_selection  import train_test_split

data = pd.read_csv("EmployeeChurn.csv")
data.columns
data.Is_Attrite.value_counts()
target_encode = {"Yes": 1, "No": 0}
data['Is_Attrite'] =  data['Is_Attrite'].map(target_encode)
final_data = pd.get_dummies(data)
x_train = final_data.drop('Is_Attrite',axis = 1)
y_train = final_data['Is_Attrite']
x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(x_train, y_train)


# Random Forest model on BOW features
from sklearn.ensemble import RandomForestClassifier
# instantiate model
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
# train model
rf.fit(x_train_split, y_train_split)
# predict on test data
rf_bow_predictions = rf.predict(x_test_split)
from sklearn.metrics import confusion_matrix, classification_report
labels = ['negative', 'positive']
print(classification_report(y_test_split, rf_bow_predictions))
pd.DataFrame(confusion_matrix(y_test_split, rf_bow_predictions), index=labels, columns=labels)


from sklearn.linear_model import LogisticRegressionCV
# L1 regularized logistic regression
lr_l1 = LogisticRegressionCV(Cs=10, cv=4, penalty='l1', solver='liblinear', max_iter=100).fit(x_train_split, y_train_split)
# predict on test data
lr_bow_predictions = lr.predict(x_test_split)
from sklearn.metrics import confusion_matrix, classification_report
labels = ['negative', 'positive']
print(classification_report(y_test_split, lr_bow_predictions))
pd.DataFrame(confusion_matrix(y_test_split, lr_bow_predictions), index=labels, columns=labels)


# L2 regularized logistic regression
lr_l2 = LogisticRegressionCV(Cs=10, cv=4, penalty='l2', solver='liblinear', max_iter=100).fit(x_train_split, y_train_split)
# predict on test data
lr_bow_predictions = lr.predict(x_test_split)
from sklearn.metrics import confusion_matrix, classification_report
labels = ['negative', 'positive']
print(classification_report(y_test_split, lr_bow_predictions))
pd.DataFrame(confusion_matrix(y_test_split, lr_bow_predictions), index=labels, columns=labels)


from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=42)
columns = x_train_split.columns
os_data_X,os_data_y=os.fit_resample(x_train_split, y_train_split)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['Is_Attrite'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['Is_Attrite']==0]))
print("Number of subscription",len(os_data_y[os_data_y['Is_Attrite']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['Is_Attrite']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['Is_Attrite']==1])/len(os_data_X))

# Random Forest model on BOW features
from sklearn.ensemble import RandomForestClassifier
# instantiate model
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
# train model
rf.fit(os_data_X, os_data_y)
# predict on test data
rf_bow_predictions = rf.predict(x_test_split)
from sklearn.metrics import confusion_matrix, classification_report
labels = ['negative', 'positive']
print(classification_report(y_test_split, rf_bow_predictions))
pd.DataFrame(confusion_matrix(y_test_split, rf_bow_predictions), index=labels, columns=labels)


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 600, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
#max_depth = [int(x) for x in np.linspace(10, 70, num = 5)]
#max_depth.append(None)
# Minimum number of samples required to split a node
#min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
#min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
#random_grid = {'n_estimators': n_estimators,
#               'max_features': max_features,
#               'max_depth': max_depth,
#               'min_samples_split': min_samples_split,
#               'min_samples_leaf': min_samples_leaf,
#               'bootstrap': bootstrap}

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'bootstrap': bootstrap}

print(random_grid)


from sklearn.ensemble import RandomForestClassifier
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=1, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(x_train_split, y_train_split)

rf_random.best_params_

best_random = rf_random.best_estimator_
base_model.fit(x_train_split, y_train_split)
rf_bow_predictions = base_model.predict(x_test_split)

from sklearn.metrics import confusion_matrix, classification_report
labels = ['negative', 'positive']
print(classification_report(y_test_split, rf_bow_predictions))
pd.DataFrame(confusion_matrix(y_test_split, rf_bow_predictions), index=labels, columns=labels)


