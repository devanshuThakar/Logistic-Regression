from cProfile import label
from secrets import token_urlsafe
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class multiclass_logistic_regression():
    def __init__(self, iterations=100, lr=0.01, lamb=0):
        self.theta = None
        self.lr = lr
        self.iterations = iterations
        self.lamb = lamb
        self.cost_list = []
        self.K = None

    def cost_function(self,X_,y):
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        y_1hot = np.zeros((y.size,y.max()+1))
        y_1hot[np.arange(y.size),y]=1
        y_1hot=torch.from_numpy(y_1hot)

        J = torch.sum(torch.sum(-1*y_1hot*logsoftmax(X_@self.theta))) + self.lamb*(torch.sum(torch.sum(self.theta*self.theta,dim=0)))
        return J
    
    def fit(self, X, y, gradient='direct'):
        self.cost_list = []
        X_ = np.append(X,np.ones((X.shape[0],1)),axis=1)
        X_=torch.from_numpy(X_)
        X_ = X_.float()  # change data type to float

        self.K = len(np.unique(y)) # K = number of classes
        
        if(gradient=='autograd'):
            self.theta = torch.rand((X_.shape[1],self.K), requires_grad=True)
            
            for iter in range(self.iterations):
                self.theta.requires_grad=True

                J = self.cost_function(X_, y)
                self.cost_list.append(J.tolist())
                J.backward()
                
                with torch.no_grad():
                    self.theta -= self.lr*torch.nan_to_num(self.theta.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    self.theta.grad.zero_()
            
            self.cost_list.append(J.tolist())
        
        else:
            self.theta = torch.rand((X_.shape[1],self.K))
            softmax = torch.nn.Softmax(dim=1)
            y_1hot = np.zeros((y.size,y.max()+1))
            y_1hot[np.arange(y.size),y]=1
            y_1hot=torch.from_numpy(y_1hot)
            for iter in range(self.iterations):
                J = self.cost_function(X_,y)
                self.cost_list.append(J.tolist())

                temp = y_1hot-(y_1hot*softmax(X_@self.theta))
                temp = temp.float()
                theta_grad = torch.t(X_)@temp + 2*self.lamb*self.theta
                self.theta = self.theta - self.lr*theta_grad
            self.cost_list.append(J.tolist())
    
    def predict(self, X, probability=False):
        softmax = torch.nn.Softmax(dim=1)
        X_ = np.append(X,np.ones((X.shape[0],1)),axis=1)
        X_=torch.from_numpy(X_)
        X_ = X_.float()  # change data type to float
        
        y_hat = (softmax(X_@self.theta)).detach().numpy()
        if(probability):
            y_hat = np.amax(y_hat,axis=1)
        else:
            y_hat =  np.argmax(y_hat, axis=1)
        return y_hat
    
    def plot_decision_boundary(self, X, y):
        xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.5,X[:,0].max()+0.2,120), np.linspace(X[:,1].min()-0.2,X[:,1].max()+0.2,120))
        
        Z = self.predict(np.vstack((xx.ravel(), yy.ravel())).T, probability=False)
        Z = Z.reshape(xx.shape)

        plt.subplot(1,1,1)
        cm_bright = ListedColormap(["#FF00FF", "#00FFFF", "#FFFF00"])
        
        plt.contourf(xx,yy,Z,cmap=cm_bright, alpha=0.4)
        scatter = plt.scatter(X[:,0],X[:,1],s=30,c=y, cmap=cm_bright, edgecolors=(0,0,0))
        legend1 = plt.legend(*scatter.legend_elements(),
                    loc="lower right", title="Classes")
        plt.title("Decision Surface")
        plt.savefig("./images/q4_decision_surface.png")
        plt.tight_layout()
        plt.show()

    def plot_cost_function(self, Fold=0):
        plt.plot(self.cost_list)
        plt.ylabel("Cost Function")
        plt.xlabel("Iteration")
        plt.title("Plot of Cost vs. iteration, lambda={}, Fold={}".format(self.lamb, Fold))
        plt.savefig("./images/q4_cost_vs_iteration_lamb_{}_fold_{}.png".format(self.lamb,Fold))
