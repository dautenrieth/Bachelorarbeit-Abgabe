# load_model.py
# Programm zur Evaluation des neuronalen Netzes
# basierend auf allen Eingabedaten
# Erstellt von Daniel Autenrieth

from numpy import loadtxt
from keras.models import load_model
import pickle
import numpy
 
# load model
model = load_model('v0.1.h5')
# summarize model.
model.summary()
# load dataset
#import data
a_file = open("input_data.pkl", "rb")
X = pickle.load(a_file)
a_file.close()

a_file = open("output_data.pkl", "rb")
y = pickle.load(a_file)
a_file.close()

X = numpy.array(X)
y = numpy.array(y)
# evaluate the model
score = model.evaluate(X, y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))