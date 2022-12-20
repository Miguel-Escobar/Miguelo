import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.integrate import nquad

# Funciones especificas del problema:
# Definir log-pdf y pdf (sin normalizar).

def logpdf(Ra, alpha):
    r1 = np.linalg.norm(Ra[0])
    r2 = np.linalg.norm(Ra[1])
    return -2*(alpha*r1 + alpha*r2)

def pdf(R, alpha):
    return np.exp(logpdf(R, alpha))

# Defino funciones para normalizar la pdf, de acuerdo con lo necesario para usar nquad:

def integral(alpha):
     #value, _ = nquad(integrable, [[-limit,limit],[-limit,limit],[-limit,limit],[-limit,limit],[-limit,limit],[-limit,limit]], args=[alpha], opts={"epsabs":10e1, "epsrel":10e1}) # Relajé mucho la tolerancia y aun así no corre
    # Al final no normalizo, no se bien por qué.
    value = 1
    return value

#def integrable(r1x,r1y,r1z,r2x,r2y,r2z, alpha):
    r1 = np.sqrt(r1x**2 + r1y**2 + r1z**2)
    r2 = np.sqrt(r1x**2 + r1y**2 + r1z**2)
    r12 = np.sqrt((r1x-r2x)**2+(r1y-r2y)**2+(r1z-r2z)**2)
    returnable = np.exp(-2*r1 - 2*r2 + .5*(r12/(1+alpha*r12)))
    return returnable

def energialocal(R, alpha):
    R1 = R[0]
    R2 = R[1]
    r1 = np.linalg.norm(R1)
    r2 = np.linalg.norm(R2)
    r12 = np.linalg.norm(R1 - R2)
    if r1 != 0 and r2 != 0 and r12 != 0:
        energy = (2*alpha-4)*(1/r1+1/r2)-2*alpha**2 + 2/r12 
        return energy
    else:
        print("invalid value replaced by zero")
        return 0

def derlog(sample, alpha):
    r1 = np.linalg.norm(sample[:,0], axis=1)
    r2 = np.linalg.norm(sample[:,1], axis=1)
    return -r1 - r2


# Funciones generales:
# Defino el paso metropolis: 

def paso_metropolis(R, alpha):    
    newR = R + np.random.normal(loc=0.0, scale=0.3, size=R.shape)
    logr = logpdf(newR, alpha) - logpdf(R, alpha)
    accept = np.log(np.random.uniform(0,1))<logr
    if accept:
        return newR, accept
    else:
        return R, accept

def METROPOLIS(ndatos, temper, initR, alpha, returnaccepted = False):
    resultados = np.zeros((ndatos,) + initR.shape)
    R = initR
    acceptance = 0
    for i in range(temper):
        R, basura = paso_metropolis(R, alpha)
    for i in range(ndatos):
        resultados[i] = R
        R, accepted = paso_metropolis(R, alpha)
        acceptance += accepted
    if returnaccepted:
        ratio = acceptance/ndatos
        return resultados, ratio
    else:
        return resultados
     
def energiaslocales(alpha, sample):
    estimacion = np.zeros(len(sample))
    for i in range(len(sample)):
        estimacion[i] = energialocal(sample[i], alpha)
    return estimacion


Ndatos = 100000
temper = 10000
R0 = np.array([[1, 1.1, .9],[-1, -1.1, -.9]]) # Inicial al azar
limit = np.inf # La integral se lleva a cabo en (-limit,limit)^6 (si es np.inf es lo mismo que integrar en R^6)
Nalphas = 40
alphas, deltaalpha = np.linspace(1,2, Nalphas, retstep=True)
EV = np.zeros(Nalphas)
Ratios = np.zeros(Nalphas)
meanderlog = np.zeros(Nalphas)
meanderlogel = np.zeros(Nalphas)
sigma2 = np.zeros(Nalphas)

for i in trange(Nalphas):
    sample, ratio = METROPOLIS(Ndatos, temper, R0, alphas[i], returnaccepted=True)
    probabilities = np.zeros(Ndatos)
    for j in range(Ndatos):
        probabilities[j] = pdf(sample[j], alphas[i])
    el = energiaslocales(alphas[i], sample)
    dlog = derlog(sample, alphas[i])
    EV[i] = np.mean(el)
    meanderlog[i] = np.mean(dlog)
    meanderlogel[i] = np.mean(dlog*el)
    Ratios[i] = ratio
    sigma2[i] = np.mean((el - EV[i])**2)

derivative = np.gradient(EV, deltaalpha)
smoothderivative = 2*(meanderlogel-meanderlog*EV)

fig = plt.figure(figsize=(12,5))
fig.clf()
axEV = fig.add_subplot(141)
axDer = fig.add_subplot(142)
axSigma = fig.add_subplot(143)
axRatio = fig.add_subplot(144)

axEV.plot(alphas, EV, label="Energía Variacional")
axEV.set_xlabel(r"$\alpha$")
axEV.set_ylabel("Energía Variacional")

axDer.plot(alphas, smoothderivative, label="Derivada suave")
axDer.plot(alphas, derivative, label="Derivada normal")
axDer.set_xlabel(r"$\alpha$")
axDer.set_ylabel(r"$\frac{dE_v}{d\alpha}$")
axDer.legend()

axSigma.plot(alphas, sigma2, label="Varianza")
axSigma.set_xlabel(r"$\alpha$")
axSigma.set_ylabel("$\sigma^2$")

axRatio.plot(alphas, Ratios, label="Fracción aceptada")
axRatio.set_xlabel(r"$\alpha$")
axRatio.set_ylabel("Fracción aceptada")

fig.tight_layout()
fig.show()

for i in range(len(smoothderivative)-1):
    if smoothderivative[i]*smoothderivative[i+1] <0:
        print("Mínimo encontrado: %.4f eV" % (((EV[i]+EV[i+1])/2)*13.6))
