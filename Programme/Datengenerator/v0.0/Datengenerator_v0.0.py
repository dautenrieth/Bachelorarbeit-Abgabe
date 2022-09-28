# Datengenerator_v0.0.py
# Programm zur Erstellung eines Datensets
# Volumenverlauf, Druckverlauf, Flow
# maschinelle Beatmung
# Erstellt von Daniel Autenrieth
import numpy as np
import matplotlib.pyplot as pyplot
import math

# inspirationzeit in sek
# expirationszeit in sek
# maschineller flow in l/sek
# Tidalvolumen in l
insp_verh = 1
insp_pause_verh = 1 #(1:1)
exp_verh = 4
exp_pause_verh = 1  #(1:2)
Abtastfrequenz = 25 #Hz
insp_zeit= 1.25
exp_zeit = 2.5
TV = 0.6
masch_flow = TV/(insp_zeit*(insp_verh/(insp_verh+insp_pause_verh)))
C = 0.1
R = 3


def Volumenverlauf(x, insp_zeit,exp_zeit, masch_flow, TV):
    x = x % (insp_zeit+exp_zeit)        # Zyklusposition bestimmen
    if x>=0 and x <= insp_zeit*(insp_verh/(insp_verh+insp_pause_verh)):
        return masch_flow*x
    elif x > insp_zeit*(insp_verh/(insp_verh+insp_pause_verh)) and x <= insp_zeit:
        return masch_flow*insp_zeit*(insp_verh/(insp_verh+insp_pause_verh))
    else:
        return np.exp(((4/(exp_zeit*(exp_verh/(exp_verh+exp_pause_verh))))*(-x+insp_zeit))+np.log(masch_flow*insp_zeit*(insp_verh/(insp_verh+insp_pause_verh))))



def Druckverlauf(x, insp_zeit,exp_zeit, masch_flow, TV, R, C):
    if x>=0 and x <= insp_zeit*(insp_verh/(insp_verh+insp_pause_verh)):
        if x <= insp_zeit*(insp_verh/(insp_verh+insp_pause_verh))/10:
            return (R*masch_flow)/(insp_zeit*(insp_verh/(insp_verh+insp_pause_verh))/10)*x      #Annährung durch Gerade
        else:
            return (masch_flow/(C*(9/10)*(insp_zeit*(insp_verh/(insp_verh+insp_pause_verh)))))*(x-(1/10)*(insp_zeit*(insp_verh/(insp_verh+insp_pause_verh))))+(R*masch_flow)

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

def exp_test(x):
    return (masch_flow/(C*(9/10)*(insp_zeit*(insp_verh/(insp_verh+insp_pause_verh)))))*(x-(1/10)*(insp_zeit*(insp_verh/(insp_verh+insp_pause_verh))))+(R*masch_flow)

print(5*(insp_zeit*(insp_verh/(insp_verh+insp_pause_verh)))+np.log(masch_flow*R))
x1=[]
druck = []
flow = []
vol = []
exp_t = []

for x in np.arange(0,exp_zeit+insp_zeit, 1/Abtastfrequenz):
    x1.append(x)
    vol.append(Volumenverlauf(x, insp_zeit, exp_zeit, masch_flow, TV))
    druck.append(Druckverlauf(x, insp_zeit,exp_zeit,masch_flow,TV,R,C))
    flow.append(Flow(x, insp_zeit, exp_zeit, masch_flow, TV))
    exp_t.append(exp_test(x))

pyplot.subplot(311)
pyplot.title('Druckverlauf')
pyplot.plot(x1,druck)
pyplot.subplot(312)
pyplot.title('Volumenverlauf')
pyplot.plot(x1,vol)
pyplot.subplot(313)
pyplot.title('Flow')
pyplot.plot(x1,flow)


# pyplot.figure(2)
# pyplot.plot(x1,flow)

# pyplot.figure(3)
# pyplot.plot(x1,exp_t)
pyplot.show()


input("Press Enter to continue...")