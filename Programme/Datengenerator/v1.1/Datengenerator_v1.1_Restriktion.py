# Datengenerator_v1.1.py
# Programm zur Erstellung eines Datensets
# Restriktion Kurve
# Erstellt von Daniel Autenrieth

import numpy as np
import matplotlib.pyplot as pyplot
import math
import scipy
from scipy import optimize

Abtastfrequenz = 25 #Hz


def exponential(x, a, k, b):
    return a*np.exp(x*k) + b#

def qubic(x, a, b, c):
    return a*x**3+b*x**2+c*x

def IVC(KG, A):
    return 6.1*KG - 0.028*A - 4.65

def restriktion(KG, A, Abtastfrequenz):

    x = []
    y = []
    y_abl = []

    # Punkte definieren
    x_A = np.array([0, 0.49, 1 ,2])
    y_A = np.array([2.5*(IVC(KG, A)/5.63), 1.11*(IVC(KG, A)/5.63), 0.49*(IVC(KG, A)/5.63), 0])

    # Funktion mit Punkten approximieren
    popt_exponential, pcov_exponential = scipy.optimize.curve_fit(exponential, x_A, y_A, p0=[1,-0.5, 1])


    a = popt_exponential[0]
    k = popt_exponential[1]
    b = popt_exponential[2]

    A2 = [[0,0,0,1], [1, 1, 1, 1], [0,0,1,0], [3, 2, 1, 0]]
    B2 = [0, 2.5*(IVC(KG, A)/5.63), 0, 0]
    erg = np.linalg.solve(A2, B2)
    a2 = erg[0]
    b2 = erg[1]
    c2 = erg[2]
    d2 = erg[3]

    for x1 in np.arange(0,3, 1/Abtastfrequenz):
        x.append(x1)
        if(x1>1):
            y.append(a*np.exp((x1-1)*k) + b)
            y_abl.append(a*k*np.exp((x1-1)*k))
        else:
            y.append(a2*x1**3 + b2*x1**2 + c2*x1 + d2)
            y_abl.append(3*a2*x1**2+2*b2*x1+c2)
    
    return x, y, y_abl

x, y, y_abl = restriktion(1.6, 25, 25)

pyplot.subplot(211)
pyplot.title('Volumen')
pyplot.plot(x,y)
pyplot.subplot(212)
pyplot.title('Fluss-Volumen')
pyplot.plot(y,y_abl)
pyplot.show()

input("Press Enter to continue...")