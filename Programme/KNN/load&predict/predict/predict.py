# Predict_mse.py
# Programm zur Evaluation von Verlaufsdaten
# Ausgabe eines Schaubilds von Klassifikation
# von Restriktion zu Normalfall
# Verwendung von statischen Daten
# Erstellt von Daniel Autenrieth
from numpy import loadtxt
from keras.models import load_model
import pickle
import numpy
import matplotlib.pyplot as pyplot
 
# load model
# Model kann zwischen MSE und Softmax variiert werden
model = load_model('v0.1.h5')
# summarize model.
model.summary()
# load dataset
#import data
a_file = open("input_data.pkl", "rb")
X = pickle.load(a_file)
a_file.close()
count = 0
data_input = []

X = numpy.array(X)

for i in X:
    norm_val = [elem/6.85 for elem in X[i]['Volumenverlauf'][0]]
    data_input.append(norm_val)
    data_input[count].append(X[i]['Parameter']['Koerpergroeße']/2)
    data_input[count].append(X[i]['Parameter']['Alter']/75)
    data_input[count].append(X[i]['Parameter']['Geschlecht'])
    count += 1

prediction_1 = []
prediction_2 = []
x = []
for a in range(1,15):
    x.append(a)

for i in data_input:
    i = numpy.array(i)
    x_pred = i.reshape(1, 78)
    pred = model.predict(x_pred)
    prediction_1.append(pred[0][0])
    prediction_2.append(pred[0][2])
    print(pred)

# Show results in plot
pyplot.figure(1)
pyplot.title('Softmax Klassifikationsergebnisse')
pyplot.plot(x, prediction_1, 'r--', label='Normal')
pyplot.plot(x, prediction_2, 'b--', label='Restriktion')
pyplot.xlabel('Beispiel')
pyplot.ylabel('Wahrscheinlichkeit')
pyplot.legend(loc='upper left')
pyplot.savefig('Softmax_Klassifikationsergebnisse.png')
pyplot.show()
