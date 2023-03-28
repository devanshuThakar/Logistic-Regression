from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
from logisticRegression import logistic_regression
import numpy as np
import pandas as pd


rng = np.random.RandomState(0)
X = rng.randn(1000,2)
Y = np.logical_xor(X[:,0] > 0, X[:,1] > 0)
Y=1*Y
x_train, x_test, y_train, y_test= train_test_split(X, Y, test_size= 0.4, random_state = 10)

# print(type(x_test))
# print(type(y_test))
train_features = pd.DataFrame(x_train)
y_train = {"Label": y_train}
train_labels = pd.DataFrame(y_train)
train_data = pd.concat([train_features, train_labels], axis=1)
# train_data = pd.DataFrame(x_train)

print(train_data)

label = 'Label'
print("Summary of class variable: \n", train_data[label].describe())

save_path = 'agModels-predictClass'  # specifies folder to store trained models
predictor = TabularPredictor(label=label, path=save_path).fit(train_data)

test_features = pd.DataFrame(x_test)
y_test = {"Label": y_test}
test_labels = pd.DataFrame(y_test)
test_data = pd.concat([test_features, test_labels], axis=1)

y_test = test_data[label]  # values to predict
test_data_nolab = test_data.drop(columns=[label])

predictor = TabularPredictor.load(save_path)  # unnecessary, just demonstrates how to load previously-trained predictor from file

y_pred = predictor.predict(test_data_nolab)
print("Predictions:  \n", y_pred)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)

print(predictor.leaderboard(test_data, silent=True))