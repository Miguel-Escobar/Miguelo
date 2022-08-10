import numpy as np
import matplotlib.pyplot as plt

ncorridas = 100
ndatos = 10**5

def esp(array):
    return np.sum(array)/len(array)

def var(array):
    return np.sum((array-esp(array))**2)/(len(array)-1)

def plot(array, name, valorreal):
    fig = plt.figure(name)
    fig.clf()    
    ax = fig.add_subplot(111)
    ax.hist(array, bins=30)

    ax.axvline(esp(array), color="r", label='Promedio')
    ax.axvline(valorreal, color="g", label= "Valor real")

    ax.legend()

    fig.show()
    print(var(array))

    return

datos = np.random.rand(ncorridas, ndatos)

esperanza = np.sum(datos, axis=1)/ndatos
limsup = np.max(datos, axis=1)
liminf = np.min(datos, axis=1)

varianza = np.zeros((ncorridas))
limsup2 = np.zeros(ncorridas)
liminf2 = np.zeros(ncorridas)

for i in np.arange(ncorridas):
    varianza[i] = (1/(ndatos-1))*(np.sum((datos[i]-esperanza[i])**2))
    limsup2[i] = esperanza[i] + esp(np.abs(datos[i]-esperanza[i]))*2
    liminf2[i] = esperanza[i] - esp(np.abs(datos[i]-esperanza[i]))*2 




