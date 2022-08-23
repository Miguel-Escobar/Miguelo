import numpy as np
import matplotlib.pyplot as plt

x = np.array([-0.15, -0.1, -0.05, 0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1])

F = np.array([[np.sin(-0.15*2*np.pi), np.sin(-0.1*2*np.pi), np.sin(-0.05*2*np.pi), np.sin(0*2*np.pi), np.sin(0.1*2*np.pi), np.sin(0.2*2*np.pi), np.sin(0.3*2*np.pi), np.sin(0.4*2*np.pi), np.sin(0.6*2*np.pi), np.sin(0.7*2*np.pi), np.sin(0.8*2*np.pi), np.sin(0.9*2*np.pi), np.sin(1*2*np.pi)],
              [np.cos(-0.15*2*np.pi), np.cos(-0.1*2*np.pi), np.cos(-0.05*2*np.pi), np.cos(0*2*np.pi), np.cos(0.1*2*np.pi), np.cos(0.2*2*np.pi), np.cos(0.3*2*np.pi), np.cos(0.4*2*np.pi), np.cos(0.6*2*np.pi), np.cos(0.7*2*np.pi), np.cos(0.8*2*np.pi), np.cos(0.9*2*np.pi), np.cos(1*2*np.pi)]]).transpose()

y = np.array([180, 150, 80, 0, -110, -180, -182, -120, 120, 190, 198, 130, 0]).transpose()

sigma = np.diag([5,6,7,2,3,10,11,10,5,6,3,7,11])

sinv = np.linalg.inv(sigma)

alpha = F.transpose() @ sinv @ F

alphainv = np.linalg.inv(F.transpose() @ sinv @ F)

b = (F.transpose() @ sinv @ y)

parametros = alphainv @ b #np.linalg.inv(F.transpose() @ sinv @ F) @ (F.transpose() @ sinv @ y)

A = parametros[0]
B = parametros[1]
plt.figure(1)
plt.plot(x, A*np.sin(2*np.pi*x) + B*np.cos(2*np.pi*x))
plt.plot(x, y)

