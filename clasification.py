from numpy.lib.function_base import append
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC



df = pd.read_csv('iris.csv') #excel en codigo as de cuenta xd 




df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width','iris_class']
#print(df.head(5))

#print(df.dtypes)

#sns.heatmap(df.corr(), annot= True, cmap='jet')
#plt.show()
train, test = train_test_split(df, test_size=0.2, random_state=30)

X_train = train[['sepal_length','sepal_width', 'petal_length', 'petal_width']]

y_train = train['iris_class']

X_test = test[['sepal_length','sepal_width', 'petal_length', 'petal_width']]

y_test = test['iris_class']

#print(X_train.shape)
#print(y_train.shape)
n_neighbors_s = []
acc_s = []
for i in list((1, 20)):

    cla = KNeighborsClassifier(n_neighbors=i)
    cla.fit(X_train, y_train)

    y_pred = cla.predict(X_test)

    acc = accuracy_score(y_pred, y_test)
    acc_s.append(acc) #tomar el valor de presiscion y lo vamos a agregar al final de la lista de presicion (acc_s)
    n_neighbors_s.append(i)

linear_svc = SVC(kernel='linear')
linear_svc.fit(X_train, y_train)
y_pred = linear_svc.predict(X_test)
acc = accuracy_score(y_pred, y_test)

print("Accouracy: %.2f" % acc)

#print(train.head(1))

#print("Accouracy: %.2f" % acc)

plt.plot(n_neighbors_s, acc_s)
plt.show()

# student performance data set 
# utilizando las columnas que creamos utiles lograr predecir la calificación final.
# de matematicas 