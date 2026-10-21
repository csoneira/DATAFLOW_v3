#%%

import time
import numpy as np

# Número de mediciones
num_measurements = 10000

# Lista para almacenar los tiempos
times = np.zeros(num_measurements)

# Realizar mediciones repetidas
for i in range(num_measurements):
    times[i] = time.time()

    # Asegurarse de que el intervalo entre mediciones sea constante
    time.sleep(0.01)  # Ajusta este valor según la precisión que necesites (10ms en este caso)

# Convertir la lista a un array de NumPy para facilitar los cálculos
times = np.array(times)

# Calcular el intervalo entre mediciones consecutivas
intervals = np.diff(times)

# Remove outliers based on quantiles
q1 = np.quantile(intervals, 0.0001)
q3 = np.quantile(intervals, 0.99)
mask = (intervals > q1) & (intervals < q3)
intervals = intervals[mask]

# Fit a gaussian to the data
from scipy.stats import norm
import matplotlib.pyplot as plt

# Calcular la media y la desviación estándar de los intervalos
mu, std = norm.fit(intervals)

# Crear un histograma de los intervalos
plt.hist(intervals, bins='auto', density=True, alpha=0.6, color='g')

# Crear un array de valores x para el ajuste
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)

# Graficar el ajuste
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)

plt.show()

# Calcular la desviación estándar de los intervalos de tiempo
std_dev = np.std(intervals)

# Imprimir los resultados
print(f"Desviación estándar de los intervalos de tiempo: {std_dev:.10f} segundos = {std_dev * 1e9:.1f} nanosegundos")
print(f"Esto indica la precisión del reloj interno del sistema.")

# %%
