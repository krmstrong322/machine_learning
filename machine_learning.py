import numpy as np
import pandas as pd
import sys

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import time
import warnings

np.set_printoptions(threshold=np.inf)
data = pd.read_csv(data file)

encoder = preprocessing.KBinsDiscretizer(n_bins=3, encode='onehot', strategy='kmeans')

#list of continuous data column heads for column transformer

#add the discrete data headers to this list
discrete_data = []

#replace "12345" with the missing value in your data
discrete_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=12345, strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

#list of continuous data column heads for column transformer

#add the continuous data headers to this list
continuous_data = []

#replace "12345" with the missing value in your data
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=12345, strategy='mean')),
    ('discretizer', encoder)])

#decides which transformer to use based on the whether the column is in the discrete or contiuous list
preprocessor = ColumnTransformer(
    transformers=[
        ('discrete', discrete_transformer, discrete_data),
        ('continuous', continuous_transformer, continuous_data)])

le = preprocessing.LabelEncoder()

class_weight = {}

classifier = RandomForestClassifier(n_estimators=200, oob_score=True, bootstrap=True, class_weight=class_weight)

selector = RFE(classifier, n_features_to_select = 10, step=0.10)

#pipeline for the GridSearchCV to use
pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                            ("RFE", selector),
                            ("classifier", classifier)])

#parameter grid for the GridSearchCV to use
param_grid = dict(RFE__n_features_to_select=[2, 3, 4, 5, 10],
                  classifier__max_depth=[2, 3, 4, 5],
                  classifier__n_estimators=[100, 200, 300, 500, 1000, 3000],
                  classifier__class_weight=[{0:1, 1:2},{0:1, 1:3}, {0:1, 1:4}, {0:1, 1:5}])

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=4, shuffle=True)


grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10, n_jobs=32, cv=5, refit=True)
grid_search.fit(X_train, y_train)
estimator = grid_search.best_estimator_
predictions = estimator.predict(X_test)
probability = estimator.predict_proba(X_test)[:,1]

#outputs csv files for the predicted output values, probability of the prediction being correct and the actual output values
y_pred = pd.DataFrame(predictions, columns=['Predictions']).to_csv('predictions.csv', index=False, header=False)
prob = pd.DataFrame(probability, columns=['Probability']).to_csv('probability.csv', index=False, header=False)
y_true = pd.DataFrame(y_test).to_csv('actual.csv', index=False, header=False)

importances = grid_search.best_estimator_.named_steps["classifier"].feature_importances_
indices = np.argsort(importances)[::-1]


print("Selected Parameters:")
selected_parameters = grid_search.best_estimator_.named_steps["RFE"].get_params()
print(selected_parameters)

print(grid_search.best_estimator_.named_steps["RFE"].support_)

print(grid_search.best_estimator_.named_steps["RFE"].ranking_)

# Print the feature ranking
print("Feature ranking:")

for f in range(X_test.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
