import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pandas import plotting
data = pd.read_csv("./giongHoa/tapDuLieuLoaiHoa.csv")
# print(data.columns)
data = data.drop("Id", axis=1)
data = data.drop("Species", axis=1)
inertia = []
K = range(1, 15)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(8,12))
plt.plot(K,inertia)
plt.xlabel("Số cụm (k)")
plt.ylabel("inertia")
plt.title("Giải thuật Elbow")
plt.grid(True)
plt.show()



# plotting.scatter_matrix(data, diagonal='hist', c=colors)
# plt.show()

# Applying KMeans with optimal clusters (3 clusters in this case)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data)

# Creating color map for scatter plot
colMap = {0: 'red', 1: 'blue', 2: 'green'}
colors = [colMap[label] for label in kmeans.labels_]

# Plotting the scatter matrix           
plotting.scatter_matrix(data, diagonal='hist', c=colors)
plt.show()