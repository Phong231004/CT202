import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
# Tải tập dữ liệu lên
data = pd.read_csv("./duBaoThoiTiet/duBaoThoiTiet.csv")
# print(data)
# print(data.columns)
# Loại bỏ thuộc tính date vì không có giá trị trong quá trình train model
data = data.drop("date", axis=1)
# Nhãn của tập dữ liệu: weather
x = data.drop("weather", axis=1)
y = data["weather"]
# print(x)
# print(y)
NB = GaussianNB()
n = 1
accuracyscore = 0 
# Dùng KFlod để phân chia ra làm 5 tập dữ liệu (lập 5 lần)
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(x):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    NB.fit(x_train, y_train)
    pred = NB.predict(x_test)
    # Tính độ chính xác
    acc = accuracy_score(y_test, pred)
    print(f"Độ chính xác lần lập {n}: {round(acc,2)*100}%")
    # Hiển thị lớp dự đoán
    for index, prediction in enumerate(pred):
        print(f"Mẫu {index+1}: Lớp dự đoán = {prediction}")
    # Hiển thị xác suất thuộc mỗi lớp 
    pred_proba = NB.predict_proba(x_test)
    print(f"Xác suất thuộc mỗi lớp: {pred_proba}")
    print("---------------------------------------")
    accuracyscore = accuracyscore + acc
    n=n+1
# Độ chính xác tổng thể của toàn bộ dữ liệu dự đoán
print(f"Độ chính xác tổng thể: accuracyscore = { round(accuracyscore/5,2)*100}%")
