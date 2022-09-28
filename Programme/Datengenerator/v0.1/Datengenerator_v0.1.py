# Datengenerator_v0.1.py
# Programm zur Erstellung eines Datensets
# Volumenverlauf, Druckverlauf, Flow
# maschinelle Beatmung
# Variierende Parameter
# Erstellt von Daniel Autenrieth

import numpy as np
import matplotlib.pyplot as pyplot
import math
import pandas as pd
from datetime import datetime
import random

def Volumenverlauf(x, insp_zeit,exp_zeit, masch_flow, TV):
    x = x % (insp_zeit+exp_zeit)        # Zyklusposition bestimmen
    if x>=0 and x <= insp_zeit*(insp_verh/(insp_verh+insp_pause_verh)):
        return masch_flow*x
    elif x > insp_zeit*(insp_verh/(insp_verh+insp_pause_verh)) and x <= insp_zeit:
        return masch_flow*insp_zeit*(insp_verh/(insp_verh+insp_pause_verh))
    else:
        return np.exp(((5/(exp_zeit*(exp_verh/(exp_verh+exp_pause_verh))))*(-x+insp_zeit))+np.log(masch_flow*insp_zeit*(insp_verh/(insp_verh+insp_pause_verh))))


def Druckverlauf(x, insp_zeit,exp_zeit, masch_flow, TV, R, C):
    if x>=0 and x <= insp_zeit*(insp_verh/(insp_verh+insp_pause_verh)):
        if x <= insp_zeit*(insp_verh/(insp_verh+insp_pause_verh))/20:
            return (R*masch_flow)/(insp_zeit*(insp_verh/(insp_verh+insp_pause_verh))/20)*x      #Annährung durch Gerade
        else:
            return (masch_flow/(C*(19/20)*(insp_zeit*(insp_verh/(insp_verh+insp_pause_verh)))))*(x-(1/20)*(insp_zeit*(insp_verh/(insp_verh+insp_pause_verh))))+(R*masch_flow)

    elif x > insp_zeit*(insp_verh/(insp_verh+insp_pause_verh)) and x <= insp_zeit:
        return np.exp(((10/(insp_zeit*(insp_verh/(insp_verh+insp_pause_verh))))*(-x+(insp_zeit*(insp_verh/(insp_verh+insp_pause_verh)))))+np.log(masch_flow*R))+masch_flow/C

    elif x > insp_zeit and x <= insp_zeit + exp_zeit*(exp_verh/(exp_verh+exp_pause_verh)):
        return np.exp((7/(exp_zeit*(exp_verh/(exp_verh+exp_pause_verh)))*(-x+insp_zeit))+np.log(masch_flow/C))
    else:
        return 0


def Flow(x, insp_zeit, exp_zeit, masch_flow, TV):
    x = x % (insp_zeit+exp_zeit)        # Ziklusposition bestimmen
    if x>=0 and x <= insp_zeit*(insp_verh/(insp_verh+insp_pause_verh)):
        return masch_flow
    elif x > insp_zeit and x <=insp_zeit+exp_zeit:
        return  -4*np.exp(((4/(exp_zeit*(exp_verh/(exp_verh+exp_pause_verh))))*(-x+insp_zeit))+np.log(masch_flow*insp_zeit*(insp_verh/(insp_verh+insp_pause_verh))))
    else:
        return 0


def save_plot(x1, druck, flow, vol, a):
    pyplot.subplot(311)
    pyplot.title('Druckverlauf')
    pyplot.plot(x1,druck)
    pyplot.subplot(312)
    pyplot.title('Volumenverlauf')
    pyplot.plot(x1,vol)
    pyplot.subplot(313)
    pyplot.title('Flow')
    pyplot.plot(x1,flow)
    pyplot.savefig('Plots/plot_'+str(a)+'.png')
    pyplot.clf()
    if a%25==0:
        print(a)

def add_0():
    x1.append(x+((exp_zeit+insp_zeit)*anzahl))
    druck.append(0)
    vol.append(Volumenverlauf(x, insp_zeit, exp_zeit, masch_flow, TV))
    flow.append(Flow(x, insp_zeit, exp_zeit, masch_flow, TV))

    values[str_name]['Volumenverlauf'].append(0)
    values[str_name]['Druckverlauf'].append(0)
    values[str_name]['Flow'].append(0)
    values[str_name]['Output'].append(0)

def add_normal():
    x1.append(x+((exp_zeit+insp_zeit)*anzahl))
    vol.append(Volumenverlauf(x, insp_zeit, exp_zeit, masch_flow, TV))
    druck.append(Druckverlauf(x, insp_zeit,exp_zeit,masch_flow,TV,R,C))
    flow.append(Flow(x, insp_zeit, exp_zeit, masch_flow, TV))

    values[str_name]['Output'].append(1)
    values[str_name]['Volumenverlauf'].append(Volumenverlauf(x, insp_zeit, exp_zeit, masch_flow, TV))
    values[str_name]['Druckverlauf'].append(Druckverlauf(x, insp_zeit,exp_zeit,masch_flow,TV,R,C))
    values[str_name]['Flow'].append(Flow(x, insp_zeit, exp_zeit, masch_flow, TV))

Verh_insp = 1
Verh_exsp = 1
# inspirationzeit in sek
# expirationszeit in sek
# maschineller flow in l/sek
# Tidalvolumen in l
insp_verh = 1
insp_pause_verh = 1 #(1:1)
exp_verh = 4
exp_pause_verh = 1  #(4:1)
Abtastfrequenz = 25 #Hz
PEEP = 0
count = 0
Anzahl_per = 4

values = {}

for Koerpergewicht in np.arange(50,100.5,1):
    TV = Koerpergewicht*0.007

    for Atemzug_min in np.arange(16,20.5,1):
        insp_zeit = 60/Atemzug_min*(Verh_insp/(Verh_insp+Verh_exsp))
        exp_zeit = 60/Atemzug_min*(Verh_exsp/(Verh_insp+Verh_exsp))
        masch_flow = TV/(insp_zeit*(insp_verh/(insp_verh+insp_pause_verh)))
        
        for C in np.arange(0.07,0.102,0.005):
            for R in np.arange(1,3.1, 0.5):
                x1=[]
                druck = []
                flow = []
                vol = []
                exp_t = []
                
                str_name = 'Set'+str(count)
                values[str_name] = {}
                values[str_name]['Volumenverlauf'] = []
                values[str_name]['Druckverlauf'] = []
                values[str_name]['Flow'] = []
                values[str_name]['Output'] = []
                values[str_name]['Parameter'] = { 'Körpergewicht' : Koerpergewicht,'TV' : TV, 'Atemzuge/min' : Atemzug_min, 'Insp': insp_zeit, 'exp' : exp_zeit, 'flow' : masch_flow, 'C' : C, 'R' : R}
                
                if count%4 == 0:
                    random.seed(datetime.now())
                    rand = random.randint(0,Anzahl_per-1)
                    random.seed(datetime.now())
                    rand2 = random.randint(0,2)

                for anzahl in np.arange(0,Anzahl_per,1):
                    
                    for x in np.arange(0,exp_zeit+insp_zeit, 1/Abtastfrequenz):
                        
                        if count%4 == 0:
                            if rand2 == 0:
                                if x >= 0 and x <= insp_zeit:
                                    add_0()
                                else:
                                    add_normal()
                            elif rand2==1:
                                if x>insp_zeit and x<= exp_zeit+insp_zeit:
                                    add_0()
                                else:
                                    add_normal()
                            else:
                                add_0()
                        else:
                            add_normal()

                count += 1
                save_plot(x1,druck,flow,vol,count)

df = pd.DataFrame.from_dict(values, orient="index")
df.to_csv("data.csv")








