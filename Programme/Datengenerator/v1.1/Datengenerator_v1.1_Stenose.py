# Datengenerator_v1.1.py
# Programm zur Erstellung eines Datensets
# Stenose Kurve
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

def stenose(KG, A, Abtastfrequenz):

    x = []
    y = []
    y_abl = []
    # Punkte definieren
    x_A = np.array([0, 0.1, 0.9 ,1])
    y_A = np.array([1.63*(IVC(KG, A)/5.63), 1.3*(IVC(KG, A)/5.63), 0.05, 0])

    # Funktion mit Punkten approximieren
    popt_exponential, pcov_exponential = scipy.optimize.curve_fit(exponential, x_A, y_A, p0=[1,-0.5, 1])
    a = popt_exponential[0]
    k = popt_exponential[1]
    b = popt_exponential[2]

    A2 = [[0,0,0,1], [1, 1, 1, 1], [0,0,1,0], [3, 2, 1, 0]]
    B2 = [0, IVC(KG, A), 0, 0]
    erg = np.linalg.solve(A2, B2)
    a2 = erg[0]
    b2 = erg[1]
    c2 = erg[2]
    d2 = erg[3]

    x_B = np.array([0, 0.25, 0.75, 1])
    y_B = np.array([IVC(KG, A), 3.55*(IVC(KG, A)/5.63), 2.96*(IVC(KG, A)/5.63), 1.63*(IVC(KG, A)/5.63)])

    A3 = [[1,1], [0, 1]]
    B3 = [1.63*(IVC(KG, A)/5.63), IVC(KG, A)]
    erg3 = np.linalg.solve(A3, B3)
    a3 = erg3[0]
    b3 = erg3[1]

    A4 = [[0.5,1], [0, 1]]
    B4 = [4.756*(IVC(KG, A)/5.63), 0.88*(IVC(KG, A)/5.63)]
    erg4 = np.linalg.solve(A4, B4)
    a4 = erg4[0]
    b4 = erg4[1]

    for x1 in np.arange(0,3, 1/Abtastfrequenz):
        x.append(x1)
        if x1 >2:
            y.append(a*np.exp((x1-2)*k) + b)
            y_abl.append(a*k*np.exp((x1-2)*k))
            
        elif(x1>=1):
            y.append(a3*(x1-1) + b3)
            y_abl.append(a3)
        
        elif x1> 0.25 and x1 < 0.75:
            y.append(a4*(x1-0.25) + b4)
            y_abl.append(a4)
            
        else:
            y.append(a2*x1**3 + b2*x1**2 + c2*x1 + d2)
            y_abl.append(3*a2*x1**2+2*b2*x1+c2)

    return x, y, y_abl


x, y, y_abl = stenose(1.6, 70, 25)

pyplot.subplot(211)
pyplot.title('Volumen')
pyplot.plot(x,y)
pyplot.subplot(212)
pyplot.title('Fluss-Volumen')
pyplot.plot(y,y_abl)
pyplot.show()

input("Press Enter to continue...")