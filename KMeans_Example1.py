import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

x = [4,5,10,4,3,11,14,6,10,12]
y = [21,19,24,17,6,25,24,22,21,21]
data = list(zip(x,y))

data = pd.DataFrame()
data['x']=x
data["y"]=y



km = KMeans(n_clusters=5)
km.fit(data)

plt.scatter(x,y, c=km.labels_)
plt.show()


