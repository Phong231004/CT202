import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,log_loss
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import numpy as np
# Tải dữ liệu
data = pd.read_csv("./duBaoThoiTiet/duBaoThoiTiet.csv")
# print(data)
# Chuyển đổi dữ liệu chuỗi về dữ liệu số nhờ LabelEncoder
label_encoder = LabelEncoder()
data["weather"] = label_encoder.fit_transform(data['weather'])
# Chuyển đổi các thuộc tính khác thành số (nếu có)
for column in data.select_dtypes(include=["object"]).columns:
    data[column] = label_encoder.fit_transform(data[column])
# print(data)
# Dùng PCA để giảm chiều (2D)
# Phân chia tập dữ liệu
data = data.drop("date", axis=1)
x = data.drop("weather", axis=1)
y = data["weather"]
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)
# print(x_pca)
# Phân chia tập dữ liệu
x_train, x_test, y_train, y_test = train_test_split(x_pca,y,test_size=0.333)
# Train model
svm = SVC(C=1000,kernel="rbf",gamma=0.0001, probability=True)
svm.fit(x_train, y_train)
pred = svm.predict(x_test)
# Tính độ chính xác
acc = accuracy_score(y_test,pred)
print(f"Độ chính xác: {round(acc*100,2)}%")
#Tính toán độ mất mát hàm Log Loss
pred_pr = svm.predict_proba(x_test)
# print(pred_pr)
logloss = log_loss(y_test, pred_pr)
print(f"Độ mất mát: {round(logloss,2)}")
# Trực quan hóa quá trình SVM
# Tạo lưới điểm để dự đoán quyết định
h = .02 #Độ phân giải cho lưới
x_min, x_max = x_train[:, 0].min() - 1, x_train[:,1].max()+1
y_min, y_max = x_train[:, 0].min()-1, x_train[:,1].max()+1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Dự đoán cho mỗi điểm trong lưới
Z = svm.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
# Vẽ biểu đồ quyết định
plt.figure(figsize=(8,6))
# Vẽ các điểm dữ liệu
plt.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap='coolwarm', s=50, edgecolors='k')
# Vẽ ranh giới quyết định
plt.contourf(xx,yy,Z,alpha=0.3,cmap='coolwarm')
# Vẽ các điểm hỗ trợ
plt.scatter(svm.support_vectors_[:,0], svm.support_vectors_[:,1], s=100, facecolors='none', edgecolors='red')
plt.title("SVM Decision Boundary with PCA (2D)")
plt.xlabel("PCA Feature 1")
plt.ylabel("PCA Feature 2")
plt.show()

# Vẽ ma trận nhầm lẫn
correlation = data.corr()
plt.figure(figsize=(8,6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()

