from audioop import mul
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.metrics import accuracy_score
from multiclassLogisticRegression import multiclass_logistic_regression
from sklearn.model_selection import StratifiedKFold
import numpy as np
from pytorch_nn import pytorch_nn


# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target
# print(len(X))
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
print("\nSplitting the train:test at 80:20 using sklearn StratifiedKFold.\n")

i = 1
iters = 10000
for train_index, test_index in skf.split(X, y):
    x_train_fold, x_test_fold = X[train_index,:], X[test_index,:]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    
    # print("Accuracy for fold {} = {}".format(i, accuracy_score(y_test_fold, y_hat)))
    print("-----------------Fold {}------------------".format(i))
    i+=1
    print("\nlambda = 0")
    mlr = multiclass_logistic_regression(iterations=iters, lr=0.0001, lamb=0)
    mlr.fit(x_train_fold, y_train_fold, gradient='autograd')
    y_hat = mlr.predict(x_test_fold)
    y_hat_train = mlr.predict(x_train_fold)
    print("The train accuracy is {0:1.4f}".format(accuracy_score(y_true=y_train_fold, y_pred=y_hat_train)))
    print("The test accuracy is {0:1.4f}".format(accuracy_score(y_true=y_test_fold, y_pred=y_hat)))
    print("The test MAE for is {0:1.5f}\n".format(np.mean(np.abs(y_test_fold - y_hat))))

    mlr.plot_cost_function(Fold=i)
    plt.clf()

    print("\nlambda = 0.5")
    mlr = multiclass_logistic_regression(iterations=iters, lr=0.0001, lamb=0.5)
    mlr.fit(x_train_fold, y_train_fold, gradient='autograd')
    y_hat = mlr.predict(x_test_fold)
    y_hat_train = mlr.predict(x_train_fold)
    print("The train accuracy is {0:1.4f}".format(accuracy_score(y_true=y_train_fold, y_pred=y_hat_train)))
    print("The test accuracy is {0:1.4f}".format(accuracy_score(y_true=y_test_fold, y_pred=y_hat)))
    print("The test MAE for is {0:1.5f}\n".format(np.mean(np.abs(y_test_fold - y_hat))))

    mlr.plot_cost_function(Fold=i)
    plt.clf()


    
mlr = multiclass_logistic_regression(iterations=10000, lr=0.0001, lamb=0)
mlr.fit(X, y, gradient='autograd')
print("Decision Surface drawn.\n")
mlr.plot_decision_boundary(X, y)


print("-------------Q4 part 6-----------")
i=1
for train_index, test_index in skf.split(X, y):
    x_train_fold, x_test_fold = X[train_index,:], X[test_index,:]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    print("-----------------Fold {}------------------".format(i))
    i+=1
    # print(y_train_fold)
    print("----------PyTorch NN with lambda 0.5 ----------")  
    nn = pytorch_nn(iterations=10000, lr=0.001,lamb=0.5,dim_input=2,dim_out=3,hidden_layer_dim=5,hidden_layers=0)
    nn.fit(x_train_fold, y_train_fold)
    y_hat = nn.predict(x_test_fold)
    y_hat_train = nn.predict(x_train_fold)

    print("The train accuracy is {0:1.4f}".format(accuracy_score(y_true=y_train_fold, y_pred=np.heaviside(y_hat_train-0.5,0))))
    print("The test accuracy is {0:1.4f}".format(accuracy_score(y_true=y_test_fold, y_pred=np.heaviside(y_hat-0.5,0))))
    print("The test MAE is {0:1.5f}\n".format(np.mean(np.abs(y_test_fold - y_hat))))
    fname = "q4_torch_cost_vs_iteration_lamb_{}_fold_{}.png".format(0.5,i)
    nn.plot_cost_function(filename=fname,Fold=i-1)

    print("----------PyTorch NN with lambda 0----------")

    nn = pytorch_nn(iterations=10000, lr=0.001,lamb=0,dim_input=2,dim_out=3,hidden_layer_dim=3,hidden_layers=0)
    nn.fit(x_train_fold, y_train_fold)
    y_hat = nn.predict(x_test_fold)
    y_hat_train = nn.predict(x_train_fold)
    # print(nn.predict(x_test))

    print("The train accuracy is {0:1.4f}".format(accuracy_score(y_true=y_train_fold, y_pred=np.heaviside(y_hat_train-0.5,0))))
    print("The test accuracy is {0:1.4f}".format(accuracy_score(y_true=y_test_fold, y_pred=np.heaviside(y_hat-0.5,0))))
    print("The test MAE is {0:1.5f}\n".format(np.mean(np.abs(y_test_fold - y_hat))))
    fname = "q4_torch_cost_vs_iteration_lamb_{}_fold_{}.png".format(0,i)
    nn.plot_cost_function(filename=fname,Fold=i-1)

nn = pytorch_nn(iterations=10000, lr=0.001,lamb=0,dim_input=2,dim_out=3,hidden_layer_dim=3,hidden_layers=0)
nn.fit(X, y)
print("Decision Surface drawn.\n")
nn.plot_decision_boundary(X, y, filename="q4_torch_decision_surface.png")
