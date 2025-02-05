import pandas as  pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
# from sklearn.tree import DecisionTreeClassifier
# Tải tập dữ liệu lên
data = pd.read_csv('./duBaoThoiTiet/duBaoThoiTiet.csv')
# print(data)
# Loại bỏ thuộc tính date vì không sử dụng trong quá trình train model
data = data.drop("date", axis=1)
# Nhãn: weather
x = data.drop("weather", axis=1)
y = data["weather"]
# print(x)
# print(y)
bagging = BaggingClassifier(n_estimators=50)
# Sử dụng kflod để phân chia tập dữ liệu và lặp 5 lần
kf = KFold(n_splits=5)
n = 0
accuracyscore = 0
for train_index, test_index in kf.split(x):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    bagging.fit(x_train, y_train)
    pred = bagging.predict(x_test)
    # Độ chính xác
    acc = accuracy_score(y_test, pred)
    print(f"Độ chính xác lần lặp {n+1}: {round(acc*100,2)}%")
    # Xác định nhãn dự đoán cho tập dữ liệu test
    for index, prediction in enumerate(pred):
        print(f"Mẫu thứ {index+1}: Nhãn dự đoán = {prediction} ")
    # Xác suất thuộc mỗi nhãn
    pred_pr = bagging.predict_proba(x_test)
    print(f"Xác suất thuộc mỗi nhãn: {pred_pr}")
    print("----------------------------------------------")
    n = n+1
    accuracyscore = accuracyscore + acc
print(f"Xác suất tổng thể: {round((accuracyscore/5)*100,2)}%")
# BaggingClassifier.__init__() got an unexpected keyword argument 'base_estimator'
# Bagging đặt giá trị mặc định của base_estimator là decisionTree

