from sklearn.cluster import KMeans
import numpy as np

class FuncionBaseRadial:

    def __init__(self, type='gaussiana', c=1):
        self.c = c
        self.type = type

    def funcion(self, x):
        if self.type == 'multicuadratica':
            return (np.sqrt(np.power(x, 2) + np.power(self.c, 2)))
        
        if self.type == 'multicuadratica_inversa':
            return (1 / np.sqrt(np.power(x, 2) + np.power(self.c, 2)))
        
        return (np.exp(- 0.5 * np.power(x / self.c, 2)))
    
class AproximacionUniversal:

    def __init__(self, funcion, reg_factor=0):
        self.funcion = funcion
        self.reg_factor = reg_factor
        self.W = None
        self.X = None

    def entrenar(self, X, d):
        n = len(X)
        self.X = X
        self.G = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self.G[i, j] = self.funcion(np.linalg.norm(self.X[i] - self.X[j]))
                
        self.W = np.linalg.solve(self.G + self.reg_factor*np.identity(n), d)


    def predecir(self, X):
        n = len(X)
        y = np.zeros(n)
        for i, x in enumerate(X):
            for w, x_i in zip(self.W, self.X):
                y[i] += w * self.funcion(np.linalg.norm(x - x_i))
        return y

class RBF():
    def __init__(self, n_clusters, funcion, reg_factor=0):
        self.n_clusters = n_clusters
        self.funcion = funcion
        self.reg_factor = reg_factor
        self.W = None

    def entrenar(self, X, y):
        kmeans = KMeans(n_clusters=self.n_clusters, n_init='auto').fit(X)
        self.centers = kmeans.cluster_centers_

        N = len(X)
        self.G = np.zeros((N, self.n_clusters))
        for i in range(N):
            for j in range(self.n_clusters):
                self.G[i, j] = self.funcion(np.linalg.norm(X[i] - self.centers[j]))
                
        self.W = np.linalg.solve(
            self.G.T.dot(self.G) + self.reg_factor*np.identity(self.n_clusters), self.G.T.dot(y))
        
    def predecir(self, X):

        N = len(X)
        phi = np.zeros((N, self.n_clusters))
        for i in range(N):
            for j in range(self.n_clusters):
                phi[i, j] = self.funcion(
                    np.linalg.norm(X[i] - self.centers[j]))
                
        return (phi.dot(self.W))