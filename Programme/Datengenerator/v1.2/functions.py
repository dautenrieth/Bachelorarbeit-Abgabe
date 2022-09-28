# functions.py
# Ausgelagerte Funktionen für Datengenerator_v1.2.py
# Erstellt von Daniel Autenrieth

import numpy as np
import matplotlib.pyplot as pyplot
import math
import scipy
from scipy import optimize

def IVC(KG, A, G):
    if G:
        return 6.1*KG - 0.028*A - 4.65
    else:
        return 4.66*KG - 0.024*A - 3.28

def exponential(x, a, k, b):
    return a*np.exp(x*k) + b

def create_set(values, count, KG, A, G):
    str_name = 'Set'+str(count)
    values[str_name] = {}
    values[str_name]['Volumenverlauf'] = []
    values[str_name]['Zeit'] = []
    values[str_name]['Flow'] = []
    values[str_name]['Output'] = []
    values[str_name]['Parameter'] = {'Koerpergroeße' : KG, 'Alter' : A, 'Geschlecht' : G}

def save_plot(count, x, y, y_abl, name):
    pyplot.subplot(211)
    pyplot.title('Volumen')
    pyplot.plot(x,y)
    pyplot.subplot(212)
    pyplot.title('Fluss-Volumen')
    pyplot.plot(y,y_abl)
    pyplot.gca().invert_yaxis()
    pyplot.gca().invert_xaxis()
    pyplot.savefig('Plots/plot_'+str(count)+'_'+str(name)+'.png')
    pyplot.clf()

def asthma(KG, A, G, Abtastfrequenz):

    x = []
    y = []
    y_abl = []

    # Punkte definieren
    x_A = np.array([0, 0.25, 1.4 ,1.5])
    y_A = np.array([IVC(KG, A, G), 3.9*(IVC(KG, A, G)/5.63), 3*(IVC(KG, A, G)/5.63), 2.95*(IVC(KG, A, G)/5.63)])

    x_B = np.array([0, 0.1, 0.5, 1.6 ,1.7])
    y_B = np.array([3.5*(IVC(KG, A, G)/5.63), 3.45*(IVC(KG, A, G)/5.63), 2.3*(IVC(KG, A, G)/5.63), 0.05, 0])

    # Funktion mit Punkten approximieren
    popt_exponential, pcov_exponential = scipy.optimize.curve_fit(exponential, x_A, y_A, p0=[1,-0.5, 1])
    a = popt_exponential[0]
    k = popt_exponential[1]
    b = popt_exponential[2]

    popt_exponential3, pcov_exponential3 = scipy.optimize.curve_fit(exponential, x_B, y_B, p0=[1,-0.5, 1])
    a3 = popt_exponential3[0]
    k3 = popt_exponential3[1]
    b3 = popt_exponential3[2]

    A2 = [[0,0,0,1], [1, 1, 1, 1], [0,0,1,0], [3, 2, 1, 0]]
    B2 = [0, IVC(KG, A, G), 0, 0]
    erg = np.linalg.solve(A2, B2)
    a2 = erg[0]
    b2 = erg[1]
    c2 = erg[2]
    d2 = erg[3]

    for x1 in np.arange(0,3, 1/Abtastfrequenz):
        x.append(x1)
        if x1 >1.35:
            y.append(a3*np.exp((x1-1.35)*k3) + b3)
            y_abl.append(a3*k3*np.exp((x1-1.35)*k3))

        elif(x1>=1):
            y.append(a*np.exp((x1-1)*k) + b)
            y_abl.append(a*k*np.exp((x1-1)*k))
            
        else:
            y.append(a2*x1**3 + b2*x1**2 + c2*x1 + d2)
            y_abl.append(3*a2*x1**2+2*b2*x1+c2)

    return x, y, y_abl

def normal(KG, A, G, Abtastfrequenz):

    x = []
    y = []
    y_abl = []
    y2 = []
    y2_abl = []

    # Punkte definieren
    x_A = np.array([0, 0.49, 1 ,2])
    y_A = np.array([IVC(KG, A, G), 2.815*(IVC(KG, A, G)/5.63), 0.955*(IVC(KG, A, G)/5.63), 0])

    # Funktion mit Punkten approximieren
    popt_exponential, pcov_exponential = scipy.optimize.curve_fit(exponential, x_A, y_A, p0=[1,-0.5, 1])


    a = popt_exponential[0]
    k = popt_exponential[1]
    b = popt_exponential[2]

    A2 = [[0,0,0,1], [1, 1, 1, 1], [0,0,1,0], [3, 2, 1, 0]]
    B2 = [0, IVC(KG, A, G), 0, 0]
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

def restriktion(KG, A, G, Abtastfrequenz):

    x = []
    y = []
    y_abl = []

    # Punkte definieren
    x_A = np.array([0, 0.49, 1 ,2])
    y_A = np.array([2.5*(IVC(KG, A, G)/5.63), 1.11*(IVC(KG, A, G)/5.63), 0.49*(IVC(KG, A, G)/5.63), 0])

    # Funktion mit Punkten approximieren
    popt_exponential, pcov_exponential = scipy.optimize.curve_fit(exponential, x_A, y_A, p0=[1,-0.5, 1])


    a = popt_exponential[0]
    k = popt_exponential[1]
    b = popt_exponential[2]

    A2 = [[0,0,0,1], [1, 1, 1, 1], [0,0,1,0], [3, 2, 1, 0]]
    B2 = [0, 2.5*(IVC(KG, A, G)/5.63), 0, 0]
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


def stenose(KG, A, G, Abtastfrequenz):

    x = []
    y = []
    y_abl = []
    # Punkte definieren
    x_A = np.array([0, 0.1, 0.9 ,1])
    y_A = np.array([1.63*(IVC(KG, A, G)/5.63), 1.3*(IVC(KG, A, G)/5.63), 0.05, 0])

    # Funktion mit Punkten approximieren
    popt_exponential, pcov_exponential = scipy.optimize.curve_fit(exponential, x_A, y_A, p0=[1,-0.5, 1])
    a = popt_exponential[0]
    k = popt_exponential[1]
    b = popt_exponential[2]

    A2 = [[0,0,0,1], [1, 1, 1, 1], [0,0,1,0], [3, 2, 1, 0]]
    B2 = [0, IVC(KG, A, G), 0, 0]
    erg = np.linalg.solve(A2, B2)
    a2 = erg[0]
    b2 = erg[1]
    c2 = erg[2]
    d2 = erg[3]

    x_B = np.array([0, 0.25, 0.75, 1])
    y_B = np.array([IVC(KG, A, G), 3.55*(IVC(KG, A, G)/5.63), 2.96*(IVC(KG, A, G)/5.63), 1.63*(IVC(KG, A, G)/5.63)])

    A3 = [[1,1], [0, 1]]
    B3 = [1.63*(IVC(KG, A, G)/5.63), IVC(KG, A, G)]
    erg3 = np.linalg.solve(A3, B3)
    a3 = erg3[0]
    b3 = erg3[1]

    A4 = [[0.5,1], [0, 1]]
    B4 = [4.756*(IVC(KG, A, G)/5.63), 0.88*(IVC(KG, A, G)/5.63)]
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

def empysem(KG, A, G, Abtastfrequenz):

    x = []
    y = []
    y_abl = []

    # Punkte definieren
    x_A = np.array([0, 0.1, 0.5, 0.9, 1])
    y_A = np.array([4.5*(IVC(KG, A, G)/5.63), 3.75*(IVC(KG, A, G)/5.63), 3.3*(IVC(KG, A, G)/5.63), 3*(IVC(KG, A, G)/5.63), 2.9*(IVC(KG, A, G)/5.63)])


    # Funktion mit Punkten approximieren
    popt_exponential, pcov_exponential = scipy.optimize.curve_fit(exponential, x_A, y_A, p0=[1,-0.5, 1])
    a = popt_exponential[0]
    k = popt_exponential[1]
    b = popt_exponential[2]

    x_B = np.array([0, 0.1, 1, 1.7, 1.8])
    y_B = np.array([3.4*(IVC(KG, A, G)/5.63), 3.3*(IVC(KG, A, G)/5.63), 1.1*(IVC(KG, A, G)/5.63), 0.05, 0])

    popt_exponential3, pcov_exponential3 = scipy.optimize.curve_fit(exponential, x_B, y_B, p0=[1,-0.5, 1])
    a3 = popt_exponential3[0]
    k3 = popt_exponential3[1]
    b3 = popt_exponential3[2]

    A2 = [[0,0,0,1], [1, 1, 1, 1], [0,0,1,0], [3, 2, 1, 0]]
    B2 = [0, 4.5*(IVC(KG, A, G)/5.63), 0, 0]
    erg = np.linalg.solve(A2, B2)
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
    
    return x, y, y_abl