#data loading
from sklearn.datasets import load_iris
myiris = load_iris()
X = myiris.data
y = myiris.target
my_feature_names = myiris.feature_names
my_target_names = myiris.target_names
print("feature names are:", my_feature_names)
print("target names are:", my_target_names)
print(X[:10])

#splitting the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#training the model
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors=3)
classifier_knn.fit(X_train, y_train)
y_pred = classifier_knn.predict(X_test)

#validate the model
from sklearn import metrics
my_accuracy = metrics.accuracy_score(y_pred, y_test)
print("Accuracy is: ", my_accuracy)
sample = [[5,5,3,2]]
my_pred = classifier_knn.predict(sample)
print(my_pred)


