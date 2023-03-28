import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

'''
class network(torch.nn.Module):
    def __init__(self, dim_input=2,dim_out=2,hidden_layers=0,hidden_layer_dim=2):
        super(network,self).__init__()
        #Creating network
        self.dim_in=dim_input
        self.dim_out=dim_out
        self.h_layers=hidden_layers
        self.h_dim=hidden_layer_dim
        self.layers=torch.nn.ModuleList()

        if(hidden_layers==0):
            self.h_dim=self.dim_out
            self.layers.append(torch.nn.Linear(self.dim_in,self.h_dim))
        elif(hidden_layers==1):
            self.layers.append(torch.nn.Linear(self.dim_in,self.h_dim))
            self.layers.append(torch.nn.Linear(self.h_dim,self.dim_out))
        else:
            self.layers.append(torch.nn.Linear(self.dim_in,self.h_dim))
            for i in range(self.h_layers-1):
                #Hidden layers having same simensions
                self.layers.append(torch.nn.Linear(self.h_dim,self.h_dim))
            self.layers.append(torch.nn.Linear(self.h_dim,self.dim_out))

    def forward(self,X):
        # sigmoid = torch.nn.Sigmoid()
        softmax = torch.nn.Softmax(dim=1)
        for i in range(self.h_layers):
            layer=self.layers[i]
            # X=sigmoid(layer(X))
            X=softmax(layer(X))
        layer=self.layers[self.h_layers]
        X=softmax(layer(X))
        return X 
'''

class network(torch.nn.Module):
    def __init__(self, dim_input=2,dim_out=2,hidden_layers=0,hidden_layer_dim=2):
        super(network,self).__init__()
        #Creating network
        self.dim_in=dim_input
        self.dim_out=dim_out
        # self.h_layers=hidden_layers
        # self.h_dim=hidden_layer_dim
        
        self.layer1=torch.nn.Linear(self.dim_in,self.dim_out)
        
    def forward(self,X):
        # sigmoid = torch.nn.Sigmoid()
        softmax = torch.nn.Softmax(dim=1)
        X=softmax(self.layer1(X))
        return X 

class pytorch_nn():
    def __init__(self, iterations=100, dim_input=2,dim_out=2,hidden_layers=0,hidden_layer_dim=2,lr=0.01,lamb=0):
        self.theta = None
        self.lr = lr
        self.iterations = iterations
        # self.lamb = lamb
        self.dim_input=dim_input
        self.dim_out=dim_out
        self.h_layers=hidden_layers
        self.h_dim=hidden_layer_dim
        self.lamb=lamb
        #Instancing the network
        self.net=network(dim_input,dim_out,hidden_layers,hidden_layer_dim)

        self.loss_list=[]

    def fit(self,X,y):
        X=torch.from_numpy(X)
        X = X.float() 
        y=torch.from_numpy(y)
        y=y.long()
        criteria=torch.nn.CrossEntropyLoss()#Criteria for loss is defined here as Cross entropy loss
        optimizer = torch.optim.SGD(params=self.net.parameters(), lr=self.lr, weight_decay=self.lamb)#Weigh_decay in SGD optimizer is L2 regularisation
        for t in range(self.iterations):
            optimizer.zero_grad()   # zero the gradient buffers
            y_pred = self.net(X)
            
            loss = criteria(y_pred, y)
            self.loss_list.append(loss.item())
            loss.backward()
            
            optimizer.step()


    def predict(self, X):
        X = torch.from_numpy(X)
        X = X.float() 
        y_pred=self.net(X)
        # y_pred=y_pred.detach().numpy()
        y_pred=torch.max(y_pred.data,1)

        y_pred=y_pred[1].detach().numpy()
        # print(y_pred)
        return y_pred

    
    def plot_decision_boundary(self, X, y,filename):
        xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.5,X[:,0].max()+0.2,120), np.linspace(X[:,1].min()-0.2,X[:,1].max()+0.2,120))
        
        Z = self.predict(np.vstack((xx.ravel(), yy.ravel())).T)
        
        Z = Z.reshape(xx.shape)

        plt.subplot(1,1,1)
        cm_bright = ListedColormap(["#FF00FF", "#00FFFF", "#FFFF00"])
        
        plt.contourf(xx,yy,Z,cmap=cm_bright, alpha=0.4)
        scatter = plt.scatter(X[:,0],X[:,1],s=30,c=y, cmap=cm_bright, edgecolors=(0,0,0))
        legend1 = plt.legend(*scatter.legend_elements(),
                    loc="lower right", title="Classes")
        plt.title("Decision Surface")
        plt.savefig("./images/"+filename)
        plt.tight_layout()
        plt.show()

    def plot_cost_function(self, filename,Fold=0):
        plt.plot(self.loss_list)
        plt.ylabel("Cost Function")
        plt.xlabel("Iteration")
        plt.title("Plot of Cost vs. iteration, lambda={}, Fold={}".format(self.lamb, Fold))
        plt.savefig("./images/"+filename)
    