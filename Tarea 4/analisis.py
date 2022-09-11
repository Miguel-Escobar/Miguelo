from importlib.metadata import distribution
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.special import gamma
numbins = 100

result = np.genfromtxt("resultados2.csv", delimiter=",")
k = result[:,0]
theta = result[:, 1]

def probability(height, array, area):
    return np.sum(array[array>=height] * area)

def heightfinder(probtarget, array, area):
    def aux(h, array, area):
        return probability(h, array, area)-probtarget
    root = root_scalar(aux, args=(array, area), x0=array.min(), x1=array.max(), xtol=0.01)
    root = root.root
    return root


fig = plt.figure("Histograma 2D")
fig.clf()
ax = fig.add_subplot(111)

alturas, kedges, thetaedges, _ = ax.hist2d(k, theta, bins=numbins, density=True)
indk, indtheta = np.unravel_index(np.argmax(alturas), alturas.shape)
deltak = kedges[1]-kedges[0]
deltatheta = thetaedges[1]-thetaedges[0]
area =deltak*deltatheta
modaconjunta = (kedges[indk]+deltak/2,thetaedges[indtheta]+deltatheta/2)
ax.plot(modaconjunta[0],modaconjunta[1] , 'o', label='Moda', color='r')
ax.set_xlabel(r"$\kappa$")
ax.set_ylabel(r"$\theta$")
ax.legend()
fig.show()

# Region de confianza 95%:

topval = 10
h95 = heightfinder(.95, alturas, area)
alturas2 = np.copy(alturas)
alturas2=np.ma.masked_array(alturas2, alturas2 < h95)
alturas2[alturas2 >= h95] = topval
X, Y = np.meshgrid(kedges, thetaedges)

fig2 = plt.figure("Region de confianza")
fig2.clf()
ax2 = fig2.add_subplot(111)
ax2.pcolormesh(X, Y, alturas2.T)
ax2.set_xlabel(r"$\kappa$")
ax2.set_ylabel(r"$\theta$")
fig2.show()

# Otra forma:

probs = [.95, .9, .68]
htop = heightfinder(probs[0], alturas, area)
hmid = heightfinder(probs[1], alturas, area)
hbot = heightfinder(probs[2], alturas, area)
alturas3 = np.copy(alturas)
alturas3[alturas3 < htop] = 0
alturas3[(alturas3 < hmid) & (alturas3 > htop)] = htop
alturas3[(alturas3 < hbot) & (alturas3 > hmid)] = hmid
alturas3[alturas > hbot] = hbot
X, Y = np.meshgrid(kedges, thetaedges)
fig4 = plt.figure("Regiones conjuntas de confianza")
fig4.clf()
ax5 = fig4.add_subplot(111)
ax5.pcolormesh(X, Y, alturas3.T)
ax5.set_xlabel(r"$\kappa$")
ax5.set_ylabel(r"$\theta$")
fig4.show()

# Histogramas marginalizados:

fig3 = plt.figure("Histogramas Marginalizados")
fig3.clf()
ax3 = fig3.add_subplot(121)
kalturas, kbordes, _ = ax3.hist(k,bins=numbins, density=True)
modak = kbordes[np.argmax(kalturas)] + (kbordes[1]-kbordes[0])*.5
ax3.axvline(modak, label='Moda', color='r')
ax3.legend()
ax3.set_xlabel(r"$\kappa$")
ax4 = fig3.add_subplot(122)
thetaalturas, thetabordes, _ = ax4.hist(theta, bins=numbins, density=True)
modatheta = thetabordes[np.argmax(thetaalturas)] + (thetabordes[1]-thetabordes[0])*.5
ax4.axvline(modatheta, label ="Moda", color='r')
ax4.legend()
ax4.set_xlabel(r"$\theta$")
fig3.show()

# Histograma datos con función por encima:

data = np.genfromtxt("energias_electrones.csv")
barridoE = np.linspace(data.min(), data.max(), 200)
def dist(E):
    k = modaconjunta[0]
    theta = modaconjunta[1]
    primertermino = 2*np.pi*np.sqrt(E)*gamma(k+1)
    segundotermino = np.pi**(3/2)*theta**(3/2)*gamma(k-1/2)*k**(3/2)
    tercertermino = (1 + E/(k*theta))**(-(k+1))
    return primertermino*tercertermino/segundotermino

fig6 = plt.figure("Distribución")
fig6.clf()
ax6 = fig6.add_subplot(111)
ax6.hist(data, bins=100, density=True)
ax6.plot(barridoE, dist(barridoE), label="distribución kappa ajustada", color='r')
ax6.set_xlabel("Energía")
ax6.set_ylabel("PDF")
ax6.legend()
fig6.show()



