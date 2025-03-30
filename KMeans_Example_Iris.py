import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd
df = sns.load_dataset("iris")
data =df.drop(['sepal_length', 'sepal_width','species'], axis=1) 

kmeans = KMeans(n_clusters=3)
lll= kmeans.fit(data)   
kkk= kmeans.fit_predict(data)  
#print("Fit Data", lll)
#print("Fit Predict Data", kkk) return index of cluster
plt.scatter(data['petal_length'], data['petal_width'],c=kmeans.labels_)
plt.show()
df['Labels'] = kmeans.labels_
#print(df)
PL = float(input("Enter Petal Length"))
PW = float(input("Enter Petal Width"))
print("Classification of Sepal = ", *kmeans.predict([[PL,PW]]))