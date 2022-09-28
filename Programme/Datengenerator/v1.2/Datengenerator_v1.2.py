# Datengenerator_v1.2.py
# Programm zur Erstellung eines Datensets
# Zusammenführung der verschiedenen Kurven
# Erstellt von Daniel Autenrieth

import numpy as np
import matplotlib.pyplot as pyplot
import math
import scipy
import pandas as pd
from scipy import optimize
import pickle
import functions as f

Abtastfrequenz = 25  # Hz


def create_set(count, KG, A, G):
    str_name = "Set" + str(count)
    values[str_name] = {}
    values[str_name]["Volumenverlauf"] = []
    values[str_name]["Zeit"] = []
    values[str_name]["Flow"] = []
    values[str_name]["Output"] = []
    values[str_name]["Parameter"] = {"Koerpergroeße": KG, "Alter": A, "Geschlecht": G}
    return str_name


def append_values(str_name, x, y, y_abl):
    values[str_name]["Volumenverlauf"].append(y)
    values[str_name]["Zeit"].append(x)
    values[str_name]["Flow"].append(y_abl)


values = {}
count = 0


# Generate Data for Male
G = 1
for KG in np.arange(1.65, 1.96, 0.01):
    for A in np.arange(25, 71, 1):

        str_name = create_set(count, KG, A, G)
        x, y, y_abl = f.normal(KG, A, G, Abtastfrequenz)
        append_values(str_name, x, y, y_abl)
        values[str_name]["Output"].append([1, 0, 0, 0, 0])
        f.save_plot(count, x, y, y_abl, "normal")
        count += 1

        str_name = create_set(count, KG, A, G)
        x, y, y_abl = f.empysem(KG, A, G, Abtastfrequenz)
        append_values(str_name, x, y, y_abl)
        values[str_name]["Output"].append([0, 1, 0, 0, 0])
        f.save_plot(count, x, y, y_abl, "empysem")
        count += 1

        str_name = create_set(count, KG, A, G)
        x, y, y_abl = f.restriktion(KG, A, G, Abtastfrequenz)
        append_values(str_name, x, y, y_abl)
        values[str_name]["Output"].append([0, 0, 1, 0, 0])
        f.save_plot(count, x, y, y_abl, "restriktion")
        count += 1

        str_name = create_set(count, KG, A, G)
        x, y, y_abl = f.stenose(KG, A, G, Abtastfrequenz)
        append_values(str_name, x, y, y_abl)
        values[str_name]["Output"].append([0, 0, 0, 1, 0])
        f.save_plot(count, x, y, y_abl, "stenose")
        count += 1

        str_name = create_set(count, KG, A, G)
        x, y, y_abl = f.asthma(KG, A, G, Abtastfrequenz)
        append_values(str_name, x, y, y_abl)
        values[str_name]["Output"].append([0, 0, 0, 0, 1])
        f.save_plot(count, x, y, y_abl, "asthma")
        count += 1

        if count % 25 == 0:
            print(count)

# Generate Data for Male
G = 0
for KG in np.arange(1.50, 1.81, 0.01):
    for A in np.arange(25, 71, 1):

        str_name = create_set(count, KG, A, G)
        x, y, y_abl = f.normal(KG, A, G, Abtastfrequenz)
        append_values(str_name, x, y, y_abl)
        values[str_name]["Output"].append([1, 0, 0, 0, 0])
        f.save_plot(count, x, y, y_abl, "normal")
        count += 1

        str_name = create_set(count, KG, A, G)
        x, y, y_abl = f.empysem(KG, A, G, Abtastfrequenz)
        append_values(str_name, x, y, y_abl)
        values[str_name]["Output"].append([0, 1, 0, 0, 0])
        f.save_plot(count, x, y, y_abl, "empysem")
        count += 1

        str_name = create_set(count, KG, A, G)
        x, y, y_abl = f.restriktion(KG, A, G, Abtastfrequenz)
        append_values(str_name, x, y, y_abl)
        values[str_name]["Output"].append([0, 0, 1, 0, 0])
        f.save_plot(count, x, y, y_abl, "restriktion")
        count += 1

        str_name = create_set(count, KG, A, G)
        x, y, y_abl = f.stenose(KG, A, G, Abtastfrequenz)
        append_values(str_name, x, y, y_abl)
        values[str_name]["Output"].append([0, 0, 0, 1, 0])
        f.save_plot(count, x, y, y_abl, "stenose")
        count += 1

        str_name = create_set(count, KG, A, G)
        x, y, y_abl = f.asthma(KG, A, G, Abtastfrequenz)
        append_values(str_name, x, y, y_abl)
        values[str_name]["Output"].append([0, 0, 0, 0, 1])
        f.save_plot(count, x, y, y_abl, "asthma")
        count += 1

        if count % 25 == 0:
            print(count)

df = pd.DataFrame.from_dict(values, orient="index")
df.to_csv("data.csv")

a_file = open("data.pkl", "wb")
pickle.dump(values, a_file)
a_file.close()
