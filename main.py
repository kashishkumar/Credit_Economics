# Credit Scoring 
"""
Objective : This program is used to assess if credit should be given to an individual or not 
based on the financial information provided.
Data : Credit history and other information taken from https://www.kaggle.com/dansbecker/aer-credit-card-data

card: Dummy variable, 1 if application for credit card accepted, 0 if not
reports: Number of major derogatory reports
age: Age n years plus twelfths of a year
income: Yearly income (divided by 10,000)
share: Ratio of monthly credit card expenditure to yearly income
expenditure: Average monthly credit card expenditure
owner: 1 if owns their home, 0 if rent
selfempl: 1 if self employed, 0 if not.
dependents: 1 + number of dependents
months: Months living at current address
majorcards: Number of major credit cards held
active: Number of active credit accounts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve

def import_data(path):
    data = pd.read_csv(path)
    #print(data.describe())
    print(data.head())
    print(data.columns)
    print(data.values.shape)
    return data
    
def visualize_data(data):
    data.hist(bins=50, figsize=(20,30))
    plt.show()
    # visualize using seaborn
    sns.pairplot(data)
    plt.show()
    # visualize using matplotlib
    data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
    plt.show()

    

def preprocess_data(data):
    # Convert owner, selfemp, card to boolean
    data['owner'] = data['owner'].map({'yes': 1, 'no': 0})
    data['selfemp'] = data['selfemp'].map({'yes': 1, 'no': 0})
    data['card'] = data['card'].map({'yes': 1, 'no': 0})
    

    # split the data into train and test using train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, data['card'], test_size=0.2, random_state=0)

    # Scale the features which are integer types using StandardScaler
    scaler = MinMaxScaler()
    columns_to_scale = ['age', 'income', 'share', 'expenditure', 'months', 'dependents', 'active', 'dependents']
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

    # Remove na values
    X_train.dropna(inplace=True)

    # Check for outliers

    # Remove outliers if that helps in the future
    print (X_train.head(20))
    print('Data Preprocessed')

    return X_train, X_test, y_train, y_test


# Build neural network using keras and tensorflow for binary classification
def build_model(X_train, activation = 'relu', neurons = [10, 10] ):
    model = keras.Sequential([
        keras.layers.Dense(neurons[0], activation=activation, input_shape=(X_train.shape[1],)),
        keras.layers.Dense(neurons[1], activation=activation),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    print('Model Built')
    return model
    
def train_model(X_train, y_train, model, optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'], epochs = 10, batch_size = 32):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    # get history of model in keras
    history = model.history
    plt.plot(history.history['accuracy'])
    plt.savefig('accuracy.png')
    print ('Model Trained')
    return model, history

def test_model(X_test, y_test, model):
    # Test the model for Xtest and ytest
    accuracy = model.evaluate(X_test, y_test)
    print('Model Tested')
    return accuracy

def predict(X_test, model):
    # Predict for X_train
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5)
    print('Model Predicted')
    return y_pred_prob, y_pred

def plot_ROC_curve(y_pred, y_test):
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('ROC_curve.png')


def plot_precision_recall_curve(y_pred, y_test):
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('Thresholds: ', thresholds)
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example: AUC={0:0.2f}'.format(auc(recall, precision)))
    plt.legend(loc="lower left")
    plt.savefig('Precision_Recall_curve.png')


def confusion_matrix_plot(y_pred, y_test):
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.savefig('confusion_matrix.png')




# 11 features and card as target

if __name__ == '__main__':
    path = '/Users/kashishkumar/Desktop/Code/Personal Projects/Credit Scoring/AER_credit_card_data.csv'
    data = import_data(path)
    #visualize_data(data)
    data.dtypes
    data.head()
    X_train, X_test, y_train, y_test= preprocess_data(data)
    model = build_model(X_train)
    model, history = train_model(X_train, y_train, model, epochs = 10, batch_size = 8)
    accuracy = test_model(X_test, y_test, model)
    print(accuracy)
    y_pred_prob, y_pred = predict(X_test, model)
    type(y_pred) == type(y_train)
    plot_ROC_curve(y_pred, y_test)
    plot_precision_recall_curve(y_pred, y_test)
    confusion_matrix_plot(y_pred, y_test)
    print('Done')


