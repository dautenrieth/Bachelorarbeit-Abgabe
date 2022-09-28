# Datengenerator_v1_var.0
# Programm zur Erstellung eines Datensatzes mit Verlauf
# Restriktion zu Normalfall
# Erstellt von Daniel Autenrieth

import numpy as np
import matplotlib.pyplot as pyplot
import math
import scipy
from scipy import optimize
import pickle

values = {}
count = 0

def exponential(x, a, k, b):
    return a*np.exp(x*k) + b#

def create_set(count, KG, A, G):
    str_name = 'Set'+str(count)
    values[str_name] = {}
    values[str_name]['Volumenverlauf'] = []
    values[str_name]['Zeit'] = []
    values[str_name]['Flow'] = []
    values[str_name]['Output'] = []
    values[str_name]['Parameter'] = {'Koerpergroeße' : KG, 'Alter' : A, 'Geschlecht' : G}
    return str_name

def append_values(str_name ,x, y, y_abl):
    values[str_name]['Volumenverlauf'].append(y)
    values[str_name]['Zeit'].append(x)
    values[str_name]['Flow'].append(y_abl)

def IVC(KG, A):
    return 6.1*KG - 0.028*A - 4.65

def restriktion(KG, A, Abtastfrequenz, var):

    x = []
    y = []
    y_abl = []

    # Punkte definieren
    x_A = np.array([0, 0.49, 1 ,2])
    y_A = np.array([2.5*(IVC(KG, A)/5.63)*var, 1.11*(IVC(KG, A)/5.63)*var, 0.49*(IVC(KG, A)/5.63)*var, 0])

    # Funktion mit Punkten approximieren
    popt_exponential, pcov_exponential = scipy.optimize.curve_fit(exponential, x_A, y_A, p0=[1,-0.5, 1])


    a = popt_exponential[0]
    k = popt_exponential[1]
    b = popt_exponential[2]

    A2 = [[0,0,0,1], [1, 1, 1, 1], [0,0,1,0], [3, 2, 1, 0]]
    B2 = [0, 2.5*(IVC(KG, A)/5.63)*var, 0, 0]
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

    # pyplot.subplot(211)
    # pyplot.title('Volumen')
    # pyplot.plot(x,y)
    # pyplot.subplot(212)
    # pyplot.title('Fluss-Volumen')
    # pyplot.plot(y,y_abl)
    # pyplot.show()
    
    return x, y, y_abl


pyplot.figure(1)



G = 1
KG = 1.8
A = 25
for var in np.arange(1, 2.4, 0.1):
    str_name = create_set(count, KG, A, G)
    x, y, y_abl = restriktion(KG, A, 25, var)
    append_values(str_name, x, y, y_abl)

    pyplot.plot(x, y, '--', label='Bsp'+str(count))

    count += 1

pyplot.title('Restriktion -> Normal')
pyplot.xlabel('Zeit in s')
pyplot.ylabel('Volumen in l')
pyplot.legend(loc = 'upper right')
pyplot.savefig('Test_zu_Norm_Bsp.png')
pyplot.show()

a_file = open("var_data.pkl", "wb")
pickle.dump(values, a_file)
a_file.close()


input("Press Enter to continue...")