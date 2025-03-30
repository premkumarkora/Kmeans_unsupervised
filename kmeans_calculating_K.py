import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd
import numpy as np

x = np.array([3,1,1,1,2,1,6,6,6,5,6,7,8,9,8,9,9,8])
y = np.array([5,4,5,6,5,8,6,7,6,7,1,2,1,2,3,2,3]) 
x= np.array(list(zip(x,y)))
inertias=[]
clusters=[]
for index in range(1,10):
    km=KMeans(n_clusters=index)
    km.fit(x)
    inertias.append(km.inertia_)
    clusters.append(index)
print(inertias)

plt.plot(clusters, inertias)
plt.show()