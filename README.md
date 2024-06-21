# Radial Basis Functions

In this repository there are an implementation of Radial Basis Functions and Universal approximator.

## Radial Basis Function Class

### Parameters:
- n_clusters (int): Number of clusters.
- funcion (string): Radial function to be used. It can be 'gaussiana', 'multicuadratica' and 'multicuadratica_inversa'.
- max_epocas (int): Maximum number of epochs.
- reg_factor (float): Regulation factor.

### Methods_
- entrenar (fit).
  Parameters:
    - X (array[array[float]]): Training input data.
    - y (array[int]): Expected output values.
- predecir (predict).
  Parameters:
    - X (array[array[float]]): Data to be classified.
  Returns:
    - An array with the classifications of the given data.

### Requirements:
- Python.
- numpy module.
 
### How to install this module
First, clone this repository with the git command below:
```
git clone https://github.com/PanquecaFuriosa/Radial-Basis-Function
```

### Examples
```
from .modelo import RBF

rbf = RBF(n_clusters = 50, funcion='gaussiana', max_epocas=10000, reg_factor=0.01)
rbf.entrenar(X, y)
Y_pred = rbf.predecir(X)
```

## Universal Approximator Class

### Parameters:
- funcion (string): Radial function to be used. It can be 'gaussiana', 'multicuadratica' and 'multicuadratica_inversa'.
- reg_factor (float): Regulation factor.

### Methods:
- entrenar (fit).
  Parameters:
    - X (array[array[float]]): Training input data.
    - d (array[int]): Expected output values.
- predecir (predict).
  Parameters:
    - X (array[array[float]]): Data to be classified.
  Returns:
    - An array with the classifications of the given data.

### Requirements:
- Python.
- numpy module.
- scikit-learn
  
### How to install this module
First, clone this repository with the git command below:
```
git clone https://github.com/PanquecaFuriosa/Radial-Basis-Function
```

### Examples
```
from .modelo import AproximacionUniversal

aprox = AproximacionUniversal(funcion='gaussiana', reg_factor=0.01)
aprox.entrenar(X, y)
Y_pred = aprox.predecir(X)
```
