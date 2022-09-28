# Predict_mse.py
# Programm zur Evaluation von Verlaufsdaten
# Ausgabe eines Schaubilds von Klassifikation
# von Restriktion zu Normalfall
# Veränderung des KNN-Models
# Erstellt von Daniel Autenrieth

from numpy import loadtxt
from keras.models import load_model
import matplotlib.pyplot as pyplot
import pickle
import numpy
 
# load model
model = load_model('model_mse_1layer.h5')
# summarize model.
model.summary()
# load dataset
#import data
a_file = open("var_data.pkl", "rb")
X = pickle.load(a_file)
a_file.close()
count = 0
data_input = []
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
pyplot.title('MSE Klassifikationsergebnisse')
pyplot.plot(x, prediction_1, 'r--', label='Normal')
pyplot.plot(x, prediction_2, 'b--', label='Restriktion')
pyplot.xlabel('Beispiel')
pyplot.ylabel('Wahrscheinlichkeit')
pyplot.legend(loc='upper left')
pyplot.savefig('MSE_Klassifikationsergebnisse.png')
pyplot.show()


