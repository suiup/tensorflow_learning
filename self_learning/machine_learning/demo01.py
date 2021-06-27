from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

print("datasets: \n", iris)
print("descr: \n",iris["DESCR"])
print("feature_name: \n",iris.feature_names)
print("feature: \n", iris.data, iris.data.shape)

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
print("test feature : \n", x_train)

