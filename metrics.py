import pandas as pd
import sys
import numpy as np
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
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import time


warnings.filterwarnings('always')

np.set_printoptions(threshold=np.inf)

#imputer if neccesary, replace 12345 with missing value in the data
impute = SimpleImputer(missing_values=12345, strategy='most_frequent')

def get_metrics(pred_file, actual_file, probability_file):
    predictions = pd.read_csv(pred_file)
    test = pd.read_csv(actual_file)
    probability = pd.read_csv(probability_file)
    f1score = metrics.f1_score(impute.fit_transform(test), impute.fit_transform(predictions), average='weighted', pos_label=None)
    fpr, tpr, threshold = metrics.roc_curve(impute.fit_transform(test), impute.fit_transform(probability), pos_label=None)
    AUC = metricss.auc(fpr, tpr)
    confusion = confusion_matrix(impute.fit_transform(test), impute.fit_transform(predictions))
    report = metrics.classification_report((impute.fit_transform(test)), impute.fit_transform(predictions))


def plot_matrix():
    labels = ['0', '1']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #other colours are available however I like Blues
    cax = ax.matshow(confusion, cmap=plt.cm.Blues)
    plt.title('Confusion matrix of the chosen predictor predictor')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_ROC():
    plt.figure(2)
    plt.plot([0, 1], [0, 1], 'k--')
    #label should be set to the name of the output variables
    plt.plot(fpr, tpr, label='Output_variables')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (Random Forest)')
    plt.legend(loc='best')
    plt.show()

get_metrics("predictions.csv", "actual.csv", "probability.csv")
plot_matrix()
plot_ROC()
