import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data = pd.read_csv("./linear regression/Salary_dataset.csv")
# print(data)
data = data.drop("STT", axis=1)
x = data.drop("Salary",axis=1)
y = data["Salary"]
linearRegression = LinearRegression()
linearRegression.fit(x,y)
pred = linearRegression.predict(x)
# Trực quan hóa các điểm lên tọa độ của biểu đồ xem nó có tuyến tính không
plt.figure(figsize=(8,12))
plt.scatter(x,y,color="blue")
plt.plot(x,pred,label = "Regression Line", color="red")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.title("LinearRegression of Salary")
plt.grid(True)
plt.show()
# print(pred)
# Vẽ đường thẳng  sau khi train model
# Hiển thị tham số chặn (w0) và các tham số (wi)
print(f"Tham số w0 = {linearRegression.intercept_}")
print(f"Tham số w1 = {linearRegression.coef_}")
