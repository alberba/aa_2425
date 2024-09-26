import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from Adaline import Adaline
from sklearn.svm import SVC


# Generació del conjunt de mostres
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=2,
                           random_state=9)
y[y == 0] = -1  # Per simplificar càlculs. En lloc de tenir les classes [1,0] tindrem [1,-1]

# Els dos algorismes es beneficien d'estandaritzar les dades, per tant, ho farem.
scaler = StandardScaler()
X_transformed = scaler.fit_transform(X)

# Entrenam un perceptron
perceptron = Adaline(eta=0.5, n_iter=60)
perceptron.fit(X_transformed, y)
y_prediction = perceptron.predict(X)


# TODO: Entrenam una SVM linear (classe SVC)
svm = SVC(C=10.0, kernel='linear')
svm.fit(X_transformed, y)

svm = SVC(kernel="linear", C=1000)
svm.fit(X_transformed, y, sample_weight=None)
# No seria necesario 
y_svc_predict = svm.predict(X)



# Mostrar resultats
plt.figure(1)
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y)

#  Mostrem els resultats Adaline, calculam un punt i la pendent de la recta del perceptron
m = -perceptron.w_[1] / perceptron.w_[2]
origen = (0, -perceptron.w_[0] / perceptron.w_[2])
plt.axline(xy1=origen, slope=m, c="blue", label="Adaline")


# TODO Mostram els resultats SVM
#Coef 0 es w1 y coef 1 es w2
# Intercept es el bias(w0)
m_svm = -svm.coef_[0][0] / svm.coef_[0][1]
origen_svm = (0, -svm.intercept_[0] / svm.coef_[0][1])

plt.axline(xy1=origen_svm, slope=m_svm, c="green", label="SVM")
print(svm.support_vectors_)


plt.scatter(x=svm.support_vectors_[:, 0], y=svm.support_vectors_[:, 1], facecolors="green", edgecolors="green")


plt.legend()
plt.show()
