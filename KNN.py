import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
#Đọc dữ liệu 
data = pd.read_csv("./duBaoThoiTiet/duBaoThoiTiet.csv")
# print(data)
# print(data.columns)
# Loại bỏ thuộc tính date vì không có ý nghĩa trong quá trình train model
data = data.drop("date", axis=1)
# print(data)
# Nhãn weather
x = data.drop("weather", axis=1)
y = data["weather"]
# print(x)
# print(y)
KNN = KNeighborsClassifier(n_neighbors=51)
# Dùng KFold để phân chia tập dữ liệu
# Phân chia ngẫu nhiên 5 lần
kf = KFold(n_splits=5)
accuracyscore = 0
n = 1
for train_index, test_index in kf.split(x):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    KNN.fit(x_train,y_train)
    pred = KNN.predict(x_test)
    # Độ chính xác
    acc = accuracy_score(y_test,pred)
    print(f"Độ chính xác lần phân chia thứ {n}: {round(acc,2)*100}%")
    # Hiện thị các lớp đã dự đoán
    for index, prediction in enumerate(pred):
        print(f"Mẫu {index+1}: Lớp dự đoán = {prediction}")
    # Xác suất thuộc mỗi lớp
    predictproba = KNN.predict_proba(x_test)
    print(f"Xác suất thuộc mỗi lớp: {predictproba}")
    print("--------------------------------------------")
    n = n+1
    accuracyscore = accuracyscore + acc
print(f"Độ chính xác tổng thể của tập dữ liệu dự đoán: {round(accuracyscore/5,2)*100}%")