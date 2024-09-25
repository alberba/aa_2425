import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Adaline import Adaline
from sklearn.svm import SVC


# Generació del conjunt de mostres
X, y = make_classification(n_samples=400, n_features=5, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1,
                           random_state=9)

# Separar les dades: train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Estandaritzar les dades: StandardScaler

# Els dos algorismes es beneficien d'estandaritzar les dades, per tant, ho farem.
scaler = StandardScaler()
X_transformed_train = scaler.fit_transform(X_train)
X_transformed_test = scaler.transform(X_test)

# Entrenam una SVM linear (classe SVC)

svm = SVC(C=1000, kernel="linear")
svm.fit(X_transformed_train, y_train, sample_weight=None)

# Prediccio
y_predict = svm.predict(X_transformed_test)


# Metrica

num_aciertos = np.sum(y_predict == y_test)

print("Número de aciertos:", num_aciertos)
print("Tasa de aciertos:", (num_aciertos / len(y_predict)))
# TODO