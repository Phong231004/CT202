import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
# Tải tập dữ liệu lên
data = pd.read_csv("./duBaoThoiTiet/duBaoThoiTiet.csv")
# print(data)
# Loại bỏ thuộc tính date vì không sử dụng cho quá trinh train model
data = data.drop("date", axis=1)
# Nhãn: weather
x = data. drop("weather", axis=1)
y = data["weather"]
# Train mô hình số lượng cây 50
RandomForest = RandomForestClassifier(n_estimators=50, criterion="gini")
n = 1
accuracyscore = 0
# Dùng Kflod để phân chia tập dữ liệu và lặp 5 lần
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(x):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    RandomForest.fit(x_train, y_train)
    pred = RandomForest.predict(x_test)
    # Độ chính xác
    acc = accuracy_score(y_test, pred)
    print(f"Độ chính xác lần lặp {n}: {round(acc,2)*100}%")
    # Xác định nhãn các dữ liệu được dự đoán trong tập test
    for index, precdiction in enumerate(pred):
        print(f"Mẫu thứ {index+1}: Nhãn dự đoán = {precdiction}")
    # Tính xác suất thuộc mỗi lớp của nhãn dự đoán
    pred_pr = RandomForest.predict_proba(x_test)
    print(f"xác xuất thuộc mỗi nhãn: {pred_pr}")
    print("----------------------------------------")
    n = n+1
    accuracyscore = accuracyscore + acc
print(f"Độ chính xác tổng thể: {round(accuracyscore/5,3)*100}%")
