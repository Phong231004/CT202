import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn import tree
# Đọc file dữ liệu
data = pd.read_csv("./duBaoThoiTiet/duBaoThoiTiet.csv")
# print(data)
# print(data.columns)
# Loại bỏ thuộc tính date vì không có ý nghĩa trong quá trình train model
data = data.drop("date", axis=1)
# Nhãn của tập dữ liệu: wether 
x = data.drop("weather", axis=1)
y = data["weather"]
# print(x)
# print(y)
deTree = DecisionTreeClassifier(criterion="entropy")
n = 1
accuracyscore = 0
# Dùng Kfold phân chia tập dữ liệu và lập 5 lần
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(x):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    deTree.fit(x_train, y_train)
    pred = deTree.predict(x_test)
    # Độ chính xác
    acc = accuracy_score(y_test,pred)
    print(f"Độ chính xác lần lặp {n}: {round(acc,2)*100}%")
    # Nhãn dự đoán
    for index, prediction in enumerate(pred):
        print(f"Thuộc tính thứ {index+1}: Nhãn dự đoán = {prediction}")
    # Xác suất thuộc mỗi lớp
    pred_pr = deTree.predict_proba(x_test)
    count =1
    for index in pred_pr:
        print(f"Xac suất mẫu {count} thuộc mỗi lớp: {index}")
        count = count+1
    # Trả về đường dẫn quyết định của các phần tử được dự đoán
    path = deTree.decision_path(x_test)
    count =1
    for index in path:
        print(f"Đường dẫn mẫu {count} đến nhãn dự đoán: {index}")
        count = count+1
    # Trả về chỉ số nút lá của các phần tử dự đoán
    apply = deTree.apply(x_test)
    count =1
    for index in apply:
        print(f"Mẫu {count} có chỉ số nút lá của các phần tử được cây dự đoán: {index}")
        count = count+1
    # Vẽ cây
    plt.figure(figsize=(12,8))
    tree.plot_tree(deTree, filled=True, feature_names=x.columns, class_names=deTree.classes_)
    plt.show()
    print("-------------------------------------------------")
    accuracyscore = accuracyscore + acc
    n = n+1
# Độ chính xác tổng thể
print(f"Độ chính xác tổng thể: {round(accuracyscore/5,2)*100}%")