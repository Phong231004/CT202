import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
# Tải tập dữ liệu
data = pd.read_csv("./duBaoThoiTiet/duBaoThoiTiet.csv")
# print(data)
# Loại bỏ thuộc tính date vì không dùng cho quá trình train model
data = data.drop("date", axis=1)
# Nhãn: weather
x = data.drop("weather", axis=1)
y = data["weather"]
# print(x)
# print(y)
n = 1
accuracyscore = 0
boosting = AdaBoostClassifier(n_estimators=50, random_state=1234)
# Dùng Kfold phân chia tập dữ liệu và lặp 5 lần
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(x):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    boosting.fit(x_train, y_train)
    pred = boosting.predict(x_test)
    # Độ chính xác
    acc = accuracy_score(y_test, pred)
    print(f"Độ chính xác lần {n}: {round(acc,2)*100}%")
    # Xác định nhãn dự đoán của mỗi dữ liệu test
    for index, precdiction in enumerate(pred):
        print(f"Mẫu thư {index+1}: Nhãn dự đoán = {precdiction}")
    # Xác suất thuộc mỗi nhãn
    pred_pr = boosting.predict_proba(x_test)
    print(f"Xác suất thuộc mỗi nhãn: {pred_pr}")
    print("------------------------------------------------")
    n = n+1
    accuracyscore = accuracyscore+acc
print(f"Độ chính xác tổng thể: {round(accuracyscore/5,2)*100}%")
# lỗi do phiên bản AdaBoosting chỉ mặc định là DiecisionTree--> sửa lại thôi
# giá trị base_estimator mặc định là Decisiontree