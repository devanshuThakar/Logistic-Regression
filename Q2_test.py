from sklearn.model_selection import train_test_split
from pytorch_nn import pytorch_nn
from sklearn.metrics import accuracy_score
import numpy as np

rng = np.random.RandomState(0)
N=1000
X = rng.randn(N,2)
Y = np.logical_xor(X[:,0] > 0, X[:,1] > 0)
Y=1*Y
print("{} points are generated randomly to train logistic regression.\n".format(N))
x_train, x_test, y_train, y_test= train_test_split(X, Y, test_size= 0.4, random_state = 10)
print("Randomly split train/test as 60:40 \n")

print("-------------Q2-----------")
print("----------PyTorch NN with lambda 0.5----------")
nn = pytorch_nn(iterations=10000, lr=0.001,lamb=0.5,dim_input=2,dim_out=2,hidden_layer_dim=5,hidden_layers=0)
nn.fit(x_train, y_train)
y_hat = nn.predict(x_test)
y_hat_train = nn.predict(x_train)

print("The train accuracy is {0:1.4f}".format(accuracy_score(y_true=y_train, y_pred=np.heaviside(y_hat_train-0.5,0))))
print("The test accuracy is {0:1.4f}".format(accuracy_score(y_true=y_test, y_pred=np.heaviside(y_hat-0.5,0))))
print("The test MAE is {0:1.5f}\n".format(np.mean(np.abs(y_test - y_hat))))

nn.plot_cost_function("q2_cost_vs_iteration_lamb_0.5.png")
nn.plot_decision_boundary(X, Y,"q2_decision_surface_lamb_0.5.png")


print("-------------Q2-----------")
print("----------PyTorch NN with lambda 0----------")

nn = pytorch_nn(iterations=10000, lr=0.001,lamb=0,dim_input=2,dim_out=2,hidden_layer_dim=3,hidden_layers=0)
nn.fit(x_train, y_train)
y_hat = nn.predict(x_test)
y_hat_train = nn.predict(x_train)
# print(nn.predict(x_test))

print("The train accuracy is {0:1.4f}".format(accuracy_score(y_true=y_train, y_pred=np.heaviside(y_hat_train-0.5,0))))
print("The test accuracy is {0:1.4f}".format(accuracy_score(y_true=y_test, y_pred=np.heaviside(y_hat-0.5,0))))
print("The test MAE is {0:1.5f}\n".format(np.mean(np.abs(y_test - y_hat))))

nn.plot_cost_function("q2_cost_vs_iteration_lamb_0.png")
nn.plot_decision_boundary(X, Y,"q2_decision_surface_lamb_0.png")