print("hello buddy!")
import numpy as np



class Layer():  # Une classe
    def __init__(self, nb_entree, nb_sortie):
        self.n_entree = nb_entree
        self.n_sortie = nb_sortie
        self.A = np.random.randn(nb_sortie, nb_entree)
        self.b = np.random.randn(nb_sortie)

    def activation(self, u):
        return np.arctan(u)

    def forward(self, X):  # calcul de activation(Ax+b)
        Y = np.matmul(self.A, X) + np.outer(self.b, np.ones(X.shape[1]))
        return self.activation(Y)

    def deriv_activation(self, u):
        return 1 / (1 + (u) ** 2)

    def backward(self, X, gx):  # retropropagation du gradient sur la couche
        temp1 = np.matmul(self.A, X)
        temp2 = self.deriv_activation(temp1 + np.outer(self.b, np.ones(X.shape[1])))
        M = np.multiply(temp2, gx)
        ge = np.matmul(np.matrix.transpose(self.A), M)
        ga = np.matmul(M, np.matrix.transpose(X))
        gb = np.zeros(self.n_sortie)
        for i in range(self.n_sortie):
            gb[i] = np.sum(M[i])
        return ge, ga, gb


class Network():  # Réseau de neurone qui est essentiellement une liste de Layers
    def __init__(self, layers_dim):
        self.list_layers = []
        for i in range(len(layers_dim) - 1):
            self.list_layers.append(Layer(layers_dim[i], layers_dim[i + 1]))

    def forward(self, Z):  # calcul du passage dans chaque couche
        X_list = []
        X_list.append(np.copy(Z))
        for layer in self.list_layers:
            Z = layer.forward(Z)
            X_list.append(np.copy(Z))

        return X_list

    def backward(self, X_list, gx):  # retropropagation globale du gradient
        list_grad = []

        for (layer, i) in zip(reversed(self.list_layers), reversed(range(len(X_list)))):
            (gx, ga, gb) = layer.backward(X_list[i - 1], gx)
            list_grad.append((ga, gb))
        return list(reversed(list_grad))