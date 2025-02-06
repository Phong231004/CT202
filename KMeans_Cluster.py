import pandas as pd
from pandas import plotting
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#  Tải dữ liệu lên
data = pd.read_csv("./giongHoa/tapDuLieuLoaiHoa.csv")
# print(data.columns)
# Loại bỏ thuộc tính Id và nhãn : Species
data = data.drop("Id", axis=1)
data = data.drop("Species", axis=1)
# print(data.columns)
# Dùng biểu đồ Elbow để tìm số cụm tối ưu cho thuật toán Kmeans
inertia = []
K = range(1,15)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    inertia.append(kmeans.inertia_)
# Vẽ đồ thị
plt.figure(figsize=(8,12))
plt.plot(K,inertia, marker='o')
plt.xlabel("Số cụm (k)")
plt.ylabel("Inertia")
plt.title("Boeeir đồ Elbow số cụm tối ưu")
plt.grid()
plt.show()
#  Số cụm tối ưu k = 3
#  Kmeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data)
# Hiển thị kết quả của quá trình gom cụm trên biểu đồ seatterplot 2 chiều
colMap = {0: 'red', 1: 'blue', 2: 'green'}
colors = [colMap[label] for label in kmeans.labels_]
plotting.scatter_matrix(data, diagonal='hist', c=colors)
# plt.title("Biểu đồ thể hiện kết quả quá trình gom cụm của Kmeans trên ma trận seatterplot 2 chiều")
plt.show()

# Trả về kết quả trên tập data
data['clusters'] = kmeans.fit_predict(data)
print(f"Kết quả trả về các nhóm của tập dữ liệu: {data['clusters']}")
# Lấy tọa độ tâm mỗi cụm
centroids = kmeans.cluster_centers_
for index, center in enumerate(centroids):
    print(f"Tâm cụm {index+1}: {center}")
