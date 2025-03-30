import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd
df = sns.load_dataset("iris")
data =df.drop(['sepal_length', 'sepal_width','species'], axis=1) 


inertias=[]
clusters=[]
for index in range(1,10):
    kmeans = KMeans(n_clusters=index)
    kmeans.fit(data)  
    inertias.append(kmeans.inertia_)
    clusters.append(index)

 
#plt.scatter(data['petal_length'], data['petal_width'],c=kmeans.labels_)
#df['Labels'] = kmeans.labels_
plt.plot(clusters, inertias)
plt.show()