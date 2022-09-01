#Datos de rayos gamma

import numpy as np
import matplotlib.pyplot as plt

# Energías
energia = np.array([0.2, 0.26, 0.32, 0.37, 0.43, 0.54, 0.68, 0.8, 1.0, 1.2, 1.4, 1.8,
                    2.1, 2.6, 3.1, 3.8, 4.6, 5.5, 6.9, 8.2, 10., 12., 14., 16.])
# Flujos
flujo_e2 = np.array([3.3, 3.15, 3.04, 2.7, 2.85, 2.65, 2.9, 2.75, 2.8, 2.65, 2.2,
                     2.0, 2.0, 1.95, 1.5, 1.4, 1.3, 1.0, 0.9, 1.1, 0.56, 0.54, 0.61, 0.33])

# Errores estándar
sigma = np.array([0.15, 0.12, 0.13, 0.14, 0.13, 0.13, 0.14, 0.13, 0.13, 0.15, 0.18,
                  0.2, 0.2, 0.22, 0.21, 0.19, 0.21, 0.18, 0.21, 0.2, 0.25, 0.3, 0.27, 0.16])


# Gráfico en log-log
plt.errorbar(energia, flujo_e2, color='red', marker='o', linestyle='None', yerr=sigma)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$E$ (TeV)", fontsize=12)
plt.ylabel("flux ($10^{-12}$TeV cm$^{-2}$s$^{-1}$)", fontsize=12)
plt.savefig("Flujo_rayosgamma.pdf")
plt.show()