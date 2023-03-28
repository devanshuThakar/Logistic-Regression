import numpy as np
import matplotlib.pyplot as plt

xx, yy = np.meshgrid(np.linspace(-3,3,50), np.linspace(-3,3,50))
rng = np.random.RandomState(0)
X = rng.randn(150,2)
Y = np.logical_xor(X[:,0] > 0, X[:,1] > 0)

plt.subplot(1,1,1)
plt.scatter(X[:,0],X[:,1],s=30,c=Y, cmap=plt.cm.Paired, edgecolors=(0,0,0))
plt.axis([-3,3,-3,3])
plt.grid(True, which='both')

plt.title('XOR Dataset')
plt.savefig('./Images/XOR_Dataset.png')
plt.show()