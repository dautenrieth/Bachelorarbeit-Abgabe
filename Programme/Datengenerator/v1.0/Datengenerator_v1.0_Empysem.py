# Datengenerator_v1.0.py
# Programm zur Erstellung eines Datensets
# Emphysem Kurve
# Erstellt von Daniel Autenrieth

import numpy as np
import matplotlib.pyplot as pyplot
import math
import scipy
from scipy import optimize

Abtastfrequenz = 25 #Hz

x = []
y = []
y_abl = []
y2 = []
y2_abl = []

def exponential(x, a, k, b):
    return a*np.exp(x*k) + b#

def qubic(x, a, b, c):
    return a*x**3+b*x**2+c*x

def square(x, a, b, c):
    return a*x**2+b*x+c

# Punkte definieren
x_A = np.array([0, 0.1, 0.5, 0.9, 1])
y_A = np.array([4.5, 3.75, 3.3, 3, 2.9])


# Funktion mit Punkten approximieren
popt_exponential, pcov_exponential = scipy.optimize.curve_fit(exponential, x_A, y_A, p0=[1,-0.5, 1])
a = popt_exponential[0]
k = popt_exponential[1]
b = popt_exponential[2]

x_B = np.array([0, 0.1, 1, 1.7, 1.8])
y_B = np.array([3.4, 3.3, 1.1, 0.05, 0])

popt_exponential3, pcov_exponential3 = scipy.optimize.curve_fit(exponential, x_B, y_B, p0=[1,-0.5, 1])
a3 = popt_exponential3[0]
k3 = popt_exponential3[1]
b3 = popt_exponential3[2]

A = [[0,0,0,1], [1, 1, 1, 1], [0,0,1,0], [3, 2, 1, 0]]
B = [0, 4.5, 0, 0]
erg = np.linalg.solve(A, B)
a2 = erg[0]
b2 = erg[1]
c2 = erg[2]
d2 = erg[3]

for x1 in np.arange(0,3, 1/Abtastfrequenz):
    x.append(x1)
    if x1 >=1.2:
        y.append(a3*np.exp((x1-1.2)*k3) + b3)
        y_abl.append(a3*k3*np.exp((x1-1.2)*k3))

    elif(x1>=1):
        y.append(a*np.exp((x1-1)*k) + b)
        y_abl.append(a*k*np.exp((x1-1)*k))
        
    else:
        y.append(a2*x1**3 + b2*x1**2 + c2*x1 + d2)
        y_abl.append(3*a2*x1**2+2*b2*x1+c2)


pyplot.subplot(211)
pyplot.title('Volumen')
pyplot.plot(x,y)
pyplot.subplot(212)
pyplot.title('Fluss-Volumen')
pyplot.plot(y,y_abl)
pyplot.show()

input("Press Enter to continue...")