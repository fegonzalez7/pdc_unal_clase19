# Programación de Computadores - UNAL
## Modulos importantes

## Numpy
(Intro muy incompleta)

Es una de las librerías más conocidas para realizar computación científica en Python. Provee una librería de alto rendimiento para la manipulación de arreglos en varias dimensiones, además de ofrecer herramientas sofisticadas para manipularlos.

**Traducción:** Hacer operaciones matemáticas con *list()* es bastante tedioso, un par de genios simplemente programaron el poder compatucional de *Matlab* para usar matrices en Python, y eso se llama *numpy*.

**Enlaces útiles:**
 + [Sitio Oficial](https://numpy.org/)
 + [Paquete para Python](https://pypi.org/project/numpy/)

**Instalación:**
Para instalar en todo el sistema:
```sh
pip install numpy
```

Para instalar en *notebook*:
```sh
!pip install numpy
```

### Arreglos by numpy
 + Un arreglo en numpy es una retícula de valores del **mismo tipo** indexadas por enteros no negativos.
 + El número de dimensiones (rank) y la forma (shape) del arreglo es una tupla de enteros que da el tamaño del arreglo para cada dimensión.
 + Se pueden crear arreglos de numpy desde listas de Python y acceder a los elementos con el operador subscript [].

Usando numpy (the ugly way):
```python
import numpy
```

Usando numpy (the right way):
```python
import numpy as np
```

**Ejemplos:**

```python
import numpy as np
a = np.array(list(range(1,5))) # Crea un arreglo lineal [1, 2, 3, 4]
print(type(a)) # Imprime "<class 'numpy.ndarray'>"
print(a)
print(a.shape) # Imprime "(4,)" es de tamano 4, de 1 dimension
print(a[0], a[1], a[2]) # 1, 2, 3
a[0] = -4 
print(a) # [-4, 2, 3, 4]
```

```python
import numpy as np
b = np.array([[1,2,3,5,6],[4,5,6,7,8]]) # Crea un arreglo 2-dimensional - matriz
print(b.shape)
print(b)
b[0,0] = 1590 # Subindices con mas logica
print(b)
print(b[0,0], b[0,1], b[1,0])
```

```python
import numpy as np
c = np.array([1,2,3]) 
d = np.array([4,5,6]) 
e = c + d
print(e) # Boommmmm
```

```python
import numpy as np
f = np.array([[1,2,3],[4,5,6]]) 
print(f.shape)
print(f.transpose().shape)
print(f) # Boommmmm
```

```python
import numpy as np
g = np.array([[1,2,3],[4,5,6],[7,8,9]])
h = np.array([[9,8,7],[6,5,4],[3,2,1]])
print(np.matmul(g,h))
```

**Ejercicio:** Hacer un code que genere una matriz de zeros de *n*x*n*.

### Indexación
```python
import numpy as np
# Crea un arreglo 2-dimensional con forma (3, 4)
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a.shape)
print(a)
b = a[:2, 1:3]
# El primer argumento indica las filas y el segundo las columnas
print(b)
print("------------------------")
# Si se modifica algo de b, se cambia algo de a
b[0, 0] = -11 # b[0, 0] es el mismo a[0, 1]
print(b)
print(a)
print(a[0,1]) # Imprime "-11"
```

### Tipo de dato
Numpy como Python determinan el tipo de dato basado en el valor. Sin embargo, dicho tipo tambien puede especificarse.

```python
import numpy as np
x = np.array([5, -4])
print(x.dtype)
x = np.array([1.0, 2.0])
print(x.dtype)
x = np.array([5, -4], dtype=np.int32)
print(x.dtype)
```

### Operaciones
Operaciones elemento a elemento:"
```python
import numpy as np
x = np.array([[1,2,5], [3,4,6]], dtype=np.float128)
y = np.array([[5,6,-1], [7,8,-6]], dtype=np.float128)
print("Suma:")
print(x + y)
print("-----")
print(np.add(x, y))
print("raiz cuadrada:")
print(np.sqrt(x))
```

### Linspace
Esta es la forma mas cool de crear arreglos
```python
import numpy as np
i = np.linspace(2, 3, num=10, endpoint=True, retstep=False)
print(i)
```

**Recrusos:**
 + [Numpy para data science](https://realpython.com/numpy-tutorial/)
 + [Numpy en español](https://www.freecodecamp.org/espanol/news/la-guia-definitiva-del-paquete-numpy-para-computacion-cientifica-en-python/)

## Matplotlip

+ Matplotlib es una librería para crear visualizaciones estáticas o animadas en Python.
+ Es posible graficar en un área con uno o más ejes (en términos de coordenadas x-y, theta-r, coordenadas polares, x-y-z, etc).
+ La forma más simple de crear una figura con ejes es usar el módulo pyplot.

**Traducción:** Cogieron la segunda mejor cosa que hace *Matlab* (gráficas) y la pusieron en *Python* de forma  L I B R E. 

**Instalación:**
Para instalar en todo el sistema:
```sh
pip install matplotlib
```

Para instalar en *notebook*:
```sh
!pip install matplotlib
```

Usando numpy (the ugly way):
```python
import matplotlib
```

Usando numpy (the right way):
```python
from matplotlib import pyplot as plt # Si solo se quieren hacer gráficas
```

**Ejemplo:** Gráfica sencilla.
```python
from matplotlib import pyplot as plt 
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
```

[![image.png](https://i.postimg.cc/zXhBDDQ0/image.png)](https://postimg.cc/rK8cQTMr)

```python
import numpy as np
from matplotlib import pyplot as plt 
x = np.linspace(0, 2, 50)
#print(x)
# Aun con el OO-style, usamos ".pyplot.figure" para crear la figura.
fig, ax = plt.subplots() # Crea la figura y los ejes.
ax.plot(x, x, label="linear") # Dibuja algunos datos en los ejes.
ax.plot(x, x**2, label="quadratic") # Dibuja mas datos en los ejes.
ax.plot(x, x**3, label="cubic") # ... y algunos mss.
ax.set_xlabel("x label") # Agrega un x-label a los ejes.
ax.set_ylabel("y label") # Agrega un y-label a los ejes.
ax.set_title("Simple Plot") # Agrega titulo a los ejes.
ax.legend() # Agrega una leyenda.
```
[![image.png](https://i.postimg.cc/kXRcn89z/image.png)](https://postimg.cc/vcM9h44t)

```python
from matplotlib import pyplot as plt 
names = ["group_a", "group_b", "group_c"]
values = [3.4, 50.3, 23]
plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle("Categorical Plotting")
plt.show()
```
[![image.png](https://i.postimg.cc/Btf1ZjBf/image.png)](https://postimg.cc/7JXLmYRK)

**Recrusos:**
 + [Matplotlib Guide](https://realpython.com/python-matplotlib-guide/)

## Entornos virtuales

Para ver en clase

[Enalce](https://mothergeo-py.readthedocs.io/en/latest/development/how-to/venv-win.html)