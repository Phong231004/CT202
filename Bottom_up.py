import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

# Load dữ liệu
data = pd.read_csv("./giongHoa/tapDuLieuLoaiHoa.csv")
# print(data)
# print(data.columns)
# Loại bỏ thuộc tính Id và nhãn để thành tập dữ liệu không có nhãn phù hợp giải thuật
data = data.drop("Id", axis=1)
data = data.drop("Species", axis=1)
# print(data)
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model.fit(data)
# In ra tổng số cụm
print(f"Number clusters: {model.n_clusters_}")
# Tìm khoảng cách xa nhất và ngắn nhất giữa các cụm trong tập dữ liệu
distances = model.distances_
print(f"Min: {round(distances.min(),2)}")
print(f"Max: {round(distances.max(),2)}")
# Trực quan hóa phân chia cụm
z = hierarchy.linkage(model.children_,"ward")
plt.title("Giải thuật dom cụm phân cấp Bottom_up")
dn = hierarchy.dendrogram(z)
plt.show()
# Xác định dữ liệu thuộc cụm nào
# Do mình không có dữ liệu mới, nên mình dùng dữ liệu huấn luyện thay nhé...
print(f"Nhóm dự đoán: {model.labels_}")