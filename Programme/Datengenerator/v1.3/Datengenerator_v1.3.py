# Datengenerator_v1.3.py
# Programm zur Erstellung eines Datensets
# Verschiebung der Kurven -> Laufende Daten
# Erstellt von Daniel Autenrieth

import numpy as np
import matplotlib.pyplot as pyplot
from scipy import optimize
import pickle
import functions as f

Abtastfrequenz = 25 #Hz

def create_set(count, KG, A, G, x, y, y_abl):
    str_name = 'Set'+str(count)
    values[str_name] = {}
    values[str_name]['Volumenverlauf'] = []
    values[str_name]['Zeit'] = []
    values[str_name]['Flow'] = []
    values[str_name]['Output'] = []
    values[str_name]['Parameter'] = {'Koerpergroeße' : KG, 'Alter' : A, 'Geschlecht' : G}
    values[str_name]['Volumenverlauf'].append(y)
    values[str_name]['Zeit'].append(x)
    values[str_name]['Flow'].append(y_abl)
    return str_name

    

values = {}
count = 0

# Generate Data for Male
G = 1
for KG in np.arange(1.65, 1.96, 0.01):
    for A in np.arange(25, 71, 1):

        
        x, y, y_abl = f.normal(KG, A, G , Abtastfrequenz)
        str_name = create_set(count, KG, A, G, x, y, y_abl)
        values[str_name]['Output'].append([1, 0, 0, 0, 0])
        # f.save_plot(count, x, y, y_abl, 'normal')
        count += 1

        
        x, y, y_abl = f.empysem(KG, A, G , Abtastfrequenz)
        str_name = create_set(count, KG, A, G, x, y, y_abl)
        values[str_name]['Output'].append([0, 1, 0, 0, 0])
        # f.save_plot(count, x, y, y_abl, 'empysem')
        count += 1

        
        x, y, y_abl = f.restriktion(KG, A, G , Abtastfrequenz)
        str_name = create_set(count, KG, A, G, x, y, y_abl)
        values[str_name]['Output'].append([0, 0, 1, 0, 0])
        # f.save_plot(count, x, y, y_abl, 'restriktion')
        count += 1

        
        x, y, y_abl = f.stenose(KG, A, G , Abtastfrequenz)
        str_name = create_set(count, KG, A, G, x, y, y_abl)
        values[str_name]['Output'].append([0, 0, 0, 1, 0])
        # f.save_plot(count, x, y, y_abl, 'stenose')
        count += 1

        
        x, y, y_abl = f.astma(KG, A, G , Abtastfrequenz)
        str_name = create_set(count, KG, A, G, x, y, y_abl)
        values[str_name]['Output'].append([0, 0, 0, 0, 1])
        # f.save_plot(count, x, y, y_abl, 'astma')
        count += 1

        if count % 25 == 0:
            print(count)

# Generate Data for Male
G = 0
for KG in np.arange(1.50, 1.81, 0.01):
    for A in np.arange(25, 71, 1):

        
        x, y, y_abl = f.normal(KG, A, G , Abtastfrequenz)
        str_name = create_set(count, KG, A, G, x, y, y_abl)
        values[str_name]['Output'].append([1, 0, 0, 0, 0])
        # f.save_plot(count, x, y, y_abl, 'normal')
        count += 1

        
        x, y, y_abl = f.empysem(KG, A, G , Abtastfrequenz)
        str_name = create_set(count, KG, A, G, x, y, y_abl)
        values[str_name]['Output'].append([0, 1, 0, 0, 0])
        # f.save_plot(count, x, y, y_abl, 'empysem')
        count += 1

        
        x, y, y_abl = f.restriktion(KG, A, G , Abtastfrequenz)
        str_name = create_set(count, KG, A, G, x, y, y_abl)
        values[str_name]['Output'].append([0, 0, 1, 0, 0])
        # f.save_plot(count, x, y, y_abl, 'restriktion')
        count += 1

        
        x, y, y_abl = f.stenose(KG, A, G , Abtastfrequenz)
        str_name = create_set(count, KG, A, G, x, y, y_abl)
        values[str_name]['Output'].append([0, 0, 0, 1, 0])
        # f.save_plot(count, x, y, y_abl, 'stenose')
        count += 1

        
        x, y, y_abl = f.astma(KG, A, G , Abtastfrequenz)
        str_name = create_set(count, KG, A, G, x, y, y_abl)
        values[str_name]['Output'].append([0, 0, 0, 0, 1])
        # f.save_plot(count, x, y, y_abl, 'astma')
        count += 1

        if count % 25 == 0:
            print(count)

# Erweitere Volumenverlauf durch Verschiebung der Kurve
for sets in values:
    versch = values[sets]['Volumenverlauf'][0].copy()
    for t in range(0, 3*Abtastfrequenz-1):
        versch.append(versch.pop(0))
        values[sets]['Volumenverlauf'].append(versch.copy())


a_file = open("data_versch.pkl", "wb")
pickle.dump(values, a_file)
a_file.close()


        
