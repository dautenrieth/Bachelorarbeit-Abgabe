# show_predict_examples.py
# Programm zur Darstellung der 
# Klassifikationsergebnisse
# Verwendung von statischen Daten
# Erstellt von Daniel Autenrieth

from numpy import loadtxt
from keras.models import load_model
import pickle
import numpy
import matplotlib.pyplot as pyplot
 
# load model
model = load_model('model_softmax_1layer.h5')
# summarize model.
model.summary()
# load dataset
a_file = open("../../data/data.pkl", "rb")
data = pickle.load(a_file)
a_file.close()
count = 0
data_input = []

# Umwandlung der Daten in einen verwendbaren Array
x = []
for a in range(0,75):
    x.append(a)

for i in data:
    norm_val = [elem/6.85 for elem in data[i]['Volumenverlauf'][0]]
    data_input.append(norm_val)
    data_input[count].append(data[i]['Parameter']['Koerpergroeße']/2)
    data_input[count].append(data[i]['Parameter']['Alter']/75)
    data_input[count].append(data[i]['Parameter']['Geschlecht'])
    count += 1

# Klassifizierung
def predict(number):
    c = numpy.array(data_input[number])
    x_pred = c.reshape(1, 78)
    pred = model.predict(x_pred)
    return pred

pyplot.ion()
pyplot.show()

# Main Loop
while True:
    for number in range(0, 1050):
        pred = predict(number)
        pyplot.clf()
        pyplot.figure(1)
        pyplot.subplot(211)
        pyplot.title('Volumenverlauf')
        vol = data_input[number][:-3]
        kg = data_input[number][75]
        alter = data_input[number][76]
        g = data_input[number][77]
        text = 'KG: '+ str(kg)+ 'A: '+str("{:.2f}".format(alter))+'G: '+str(g)
        pyplot.plot(x, vol, 'r--')
        pyplot.subplot(212)
        pyplot.title('MSE & ReLU Klassifikationsergebnisse')
        pyplot.bar(['Normal', 'Empysem', 'Restriktion', 'Stenose', 'Astma'], pred[0])
        pyplot.ylabel('Wahrscheinlichkeit')
        pyplot.text(1, 1, text, fontsize=12)
        pyplot.draw()
        pyplot.pause(1)