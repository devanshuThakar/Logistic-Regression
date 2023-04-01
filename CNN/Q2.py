from sklearn.model_selection import train_test_split
from logisticRegression import logistic_regression
import numpy as np
import time
from matplotlib import projections, pyplot

rng = np.random.RandomState(0)

def time_taken(N,D,LR):
    X = rng.randn(N,D)
    Y = rng.randint(0,2,size=N)

    # print("{} points are generated randomly to train logistic regression.\n".format(N))
    x_train, x_test, y_train, y_test= train_test_split(X, Y, test_size= 0.4, random_state = 10)
    # print("Randomly split train/test as 60:40 \n")
    #Train
    start=time.process_time()
    LR.fit(x_train, y_train, gradient='autograd')#Using autograd method from last assignment
    end=time.process_time()
    train_t=end-start

    start=time.process_time()
    y_hat = LR.predict(x_test)
    end=time.process_time()
    test_t=end-start

    return [train_t,test_t]

train_time=[]
test_time=[]

N_list=[900,3000,9000,30000,90000,300000]
D_list=[80,200,800,2000,8000]

LR = logistic_regression(iterations=10000, lr=0.001,lamb=0.01)

#Varying Number of samples
D0=10
for N in N_list:
    t=time_taken(N,D0,LR)
    train_time.append(t[0])
    test_time.append(t[1])

#Varying number of features
N0=5000
for D in D_list:
    t=time_taken(N0,D,LR)
    train_time.append(t[0])
    test_time.append(t[1])

# print(train_time)
# print(test_time)
pyplot.subplot(211)
pyplot.plot(N_list,train_time[0:len(N_list)])
pyplot.scatter(N_list,train_time[0:len(N_list)])
pyplot.ylabel("Time in s")
pyplot.xlabel("Number of samples")
pyplot.ylim(0,max(train_time[0:len(N_list)]))
pyplot.title("Training Time Vs N")
pyplot.subplot(212)
pyplot.plot(N_list,test_time[0:len(N_list)])
pyplot.scatter(N_list,test_time[0:len(N_list)])
pyplot.ylim(0)
pyplot.title("Testing Time Vs N")
pyplot.ylabel("Time in s")
pyplot.xlabel("Number of samples")
pyplot.tight_layout()
pyplot.savefig('images/Q2_N_plot.png')
pyplot.close()

pyplot.subplot(211)
pyplot.plot(D_list,train_time[len(N_list):])
pyplot.scatter(D_list,train_time[len(N_list):])
pyplot.ylim(0,max(train_time[len(N_list):]))
pyplot.title("Training Time Vs D")
pyplot.ylabel("Time in s")
pyplot.xlabel("Number of features")
pyplot.subplot(212)
pyplot.plot(D_list,test_time[len(N_list):])
pyplot.scatter(D_list,test_time[len(N_list):])
pyplot.ylim(0)
pyplot.title("Testing Time Vs D")
pyplot.ylabel("Time in s")
pyplot.xlabel("Number of features")
pyplot.tight_layout()
pyplot.savefig('images/Q2_D_plot.png')
pyplot.close()




