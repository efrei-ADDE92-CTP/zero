

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

iris_data = datasets.load_iris()
X = iris_data.data
y = iris_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

score = model.score(X_test, y_test)

random_index = np.random.choice(X_test.shape[0])
random_iris = X_test[random_index, :]

prediction = model.predict([random_iris])
predicted_class = iris_data.target_names[prediction][0]
real_class = iris_data.target_names[prediction][0]


print(f"Accuracy du modèle de prédiction est de : {score:.5f}")
print(f"Valeurs de l'iris sélectionnée : {random_iris}")
print(f"Prédiction de la classe de l'iris sélectionnée : {predicted_class}")

joblib.dump(model, "iris_classification_model.pkl")