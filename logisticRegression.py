from secrets import token_urlsafe
import torch
import numpy as np
import matplotlib.pyplot as plt

def cost_function(X_, y, theta, lamb):
    sigmoid = torch.nn.Sigmoid()
    z=sigmoid(X_@theta)
    J = torch.sum(torch.nan_to_num(
        -1*(y*torch.log(z) + (1-y)* torch.log(1-z)), nan=0.0, posinf=0.0, neginf=0.0)) + lamb*(torch.t(theta)@theta)

    return J

class logistic_regression():
    def __init__(self, iterations=100, lr=0.01, lamb=0):
        self.theta = None
        self.lr = lr
        self.iterations = iterations
        self.lamb = lamb
        self.cost_list = []
    
    def fit(self, X, y, gradient='direct'):
        X_ = np.append(X,np.ones((X.shape[0],1)),axis=1)
        X_=torch.from_numpy(X_)
        X_ = X_.float()  # change data type to float
        y=torch.from_numpy(y)
        y=torch.reshape(y,(y.shape[0],1))

        sigmoid = torch.nn.Sigmoid()
        
        if(gradient=='autograd'):
            theta = torch.rand((X_.shape[1],1), requires_grad=True)
            
            for iter in range(self.iterations):
                theta.requires_grad=True

                J = cost_function(X_, y, theta=theta, lamb=self.lamb)
                self.cost_list.append(J.tolist()[0])
                J.backward()
                
                with torch.no_grad():
                    theta -= self.lr*torch.nan_to_num(theta.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    theta.grad.zero_()
            
            self.cost_list.append(J.tolist()[0])
            self.theta = theta
        
        else:
            theta = torch.rand((X_.shape[1],1))
            for iter in range(self.iterations):
                J = cost_function(X_, y, theta=theta,lamb=self.lamb)
                self.cost_list.append(J.tolist()[0])

                z=sigmoid(X_@theta)               
                z_minus_y=(z-y).float()
                theta_grad = torch.t(X_)@z_minus_y + 2*self.lamb*theta
                theta = theta - self.lr*theta_grad
            self.cost_list.append(J.tolist()[0])
            self.theta = theta
    
    def predict(self, X):
        X_ = np.append(X,np.ones((X.shape[0],1)),axis=1)
        X_=torch.from_numpy(X_)
        X_ = X_.float()  # change data type to float
        
        sigmoid = torch.nn.Sigmoid()
        y_hat = sigmoid(X_@self.theta)
        y_hat = y_hat.detach().numpy()
        return y_hat
    
    def plot_decision_boundary(self, X, y):
        xx, yy = np.meshgrid(np.linspace(-3,3,50), np.linspace(-3,3,50))
        
        Z = self.predict(np.vstack((xx.ravel(), yy.ravel())).T)
        Z = Z.reshape(xx.shape)

        plt.subplot(1,1,1)
        image = plt.imshow(
            Z,
            interpolation="nearest",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            aspect="auto",
            origin="lower",
            cmap=plt.cm.PuOr_r,
        )
        plt.scatter(X[:,0],X[:,1],s=30,c=y, cmap=plt.cm.Paired, edgecolors=(0,0,0))
        contours = plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors=["k"])
        plt.axis([-3,3,-3,3])
        plt.colorbar(image)
        plt.title("Decision Surface")
        plt.savefig("./images/q1_decision_surface.png")
        plt.tight_layout()
        plt.show()

    def plot_cost_function(self):
        plt.plot(self.cost_list)
        plt.ylabel("Cost Function")
        plt.xlabel("Iteration")
        plt.title("Plot of Cost vs. iteration for lambda = {}".format(self.lamb))
        plt.savefig("./images/q1_cost_vs_iteration_lamb_{}.png".format(self.lamb))
        plt.show()
        


    # def plot_decision_boundary(self, X, y):
    #     bias = self.theta[2]
    #     weight1, weight2 = self.theta[0], self.theta[1]
    #     c = -bias/weight2
    #     m = -weight1/weight2
    #     xmin, xmax = -1, 2
    #     ymin, ymax = -1-2, 2.5+2
    #     xd = np.array([xmin-2, xmax+2])
    #     yd = m*xd + c
    #     plt.plot(xd, yd, 'k', lw=1, ls='--')
    #     plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
    #     plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

    #     plt.scatter(*X[y==0].T, s=8, alpha=0.5)
    #     plt.scatter(*X[y==1].T, s=8, alpha=0.5)
    #     plt.xlim(xmin-2, xmax+2)
    #     plt.ylim(ymin-2, ymax+2)
    #     plt.xlabel('Feature_1')
    #     plt.ylabel('Feature_2')
    #     plt.savefig("./images/q1_db.png")
    #     plt.show()