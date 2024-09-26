from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=0.5,
                           random_state=8)
# Ja no necessitem canviar les etiquetes, Scikit ho fa per nosaltres

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Els dos algorismes es beneficien d'estandaritzar les dades

scaler = MinMaxScaler() #StandardScaler()
X_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)

# Divide por el número de características y la varianza
gamma = 1.0/ (X_transformed.shape[1] * X_transformed.var())

def kernel_lineal(x1, x2):
     return x1.dot(x2.T)

def kernel_gauss(x1, x2):
     return np.exp(-gamma * (distance_matrix(x1, x2))**2)

def kernel_poly(x1, x2, degree=3):
     return (gamma * x1.dot(x2.T)) ** degree

kernels = {"linear": kernel_lineal, "rbf": kernel_gauss, "poly": kernel_poly}

for kernel in kernels.keys() :
     # Entrenamos con el kernel creado
     svm_user = SVC(C=1000, kernel=kernels[kernel])
     svm_user.fit(X_transformed, y_train, sample_weight=None)
     y_predict_user = svm_user.predict(X_test)
     print("Precision score", kernels[kernel].__name__ + ":", precision_score(y_true=y_test, y_pred=y_predict_user))
     disp = DecisionBoundaryDisplay.from_estimator(svm_user, X_transformed, response_method="predict")
     disp.ax_.scatter(X_transformed[:, 0], X_transformed[:, 1], edgecolor="k")
     plt.show()

     # Entrenamos con la función de la libreria
     svm_lib = SVC(C=1000, kernel=kernel)
     svm_lib.fit(X_transformed, y_train, sample_weight=None)
     y_predict_lib = svm_lib.predict(X_test)
     print("Precision score", kernel + ":", precision_score(y_true=y_test, y_pred=y_predict_lib))
     print("")



