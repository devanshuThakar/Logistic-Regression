from sklearn.model_selection import train_test_split
from logisticRegression import logistic_regression
from sklearn.metrics import accuracy_score
import numpy as np


rng = np.random.RandomState(0)
N=150
X = rng.randn(N,2)
Y = np.logical_xor(X[:,0] > 0, X[:,1] > 0)
Y=1*Y
print("{} points are generated randomly to train logistic regression.\n".format(N))
x_train, x_test, y_train, y_test= train_test_split(X, Y, test_size= 0.4, random_state = 10)
print("Randomly split train/test as 60:40 \n")


print("-------------Q1-1-----------")
print("------------Direct----------")
LR = logistic_regression(iterations=10000, lr=0.001)
LR.fit(x_train, y_train, gradient='simple')
y_hat = LR.predict(x_test)
y_hat_train = LR.predict(x_train)
print("The train accuracy is {0:1.4f}".format(accuracy_score(y_true=y_train, y_pred=np.heaviside(y_hat_train-0.5,0))))
print("The test accuracy is {0:1.4f}".format(accuracy_score(y_true=y_test, y_pred=np.heaviside(y_hat-0.5,0))))
print("The test MAE is {0:1.5f}.\n".format(np.mean(np.abs(y_test - y_hat))))

print("-------------Q1-2-----------")
print("----------Autograd----------")
LR = logistic_regression(iterations=10000, lr=0.001,lamb=0)
LR.fit(x_train, y_train, gradient='autograd')
y_hat = LR.predict(x_test)
y_hat_train = LR.predict(x_train)
print("The train accuracy is {0:1.4f}".format(accuracy_score(y_true=y_train, y_pred=np.heaviside(y_hat_train-0.5,0))))
print("The test accuracy is {0:1.4f}".format(accuracy_score(y_true=y_test, y_pred=np.heaviside(y_hat-0.5,0))))
print("The test MAE is {0:1.5f}\n".format(np.mean(np.abs(y_test - y_hat))))
LR.plot_cost_function()

print("-------------Q1-3-----------")
print("Cross Entropy Loss and Regularization with lambda = 0.5\n")
LR = logistic_regression(iterations=10000, lr=0.001, lamb=0.5)
LR.fit(x_train, y_train, gradient='simple')
y_hat = LR.predict(x_test)
y_hat_train = LR.predict(x_train)
print("The train accuracy is {0:1.4f}".format(accuracy_score(y_true=y_train, y_pred=np.heaviside(y_hat_train-0.5,0))))
print("The test accuracy is {0:1.4f}".format(accuracy_score(y_true=y_test, y_pred=np.heaviside(y_hat-0.5,0))))
print("The test MAE is {0:1.5f}.\n".format(np.mean(np.abs(y_test - y_hat))))
LR.plot_cost_function()

print("-------------Q1-5-----------")
print("Decision Surface drawn.\n")
LR.plot_decision_boundary(X, Y)