# train_nn_v3_mse_sweep_one_layer_size_change.py
# Trainingsprogramm
# 1 Hidden-Layers, ReLU, Optimierervariation
# Veränderbare laufende Daten -> Parameterübergabe
# Erstellt von Daniel Autenrieth

import argparse
import os
import pickle
import numpy

from keras import backend as K
from keras import optimizers
from keras.callbacks import Callback
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, InputLayer
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

import wandb
from wandb.keras import WandbCallback

import sklearn
from sklearn.model_selection import train_test_split

print(sklearn.__version__)

labels =["Normal", "Empysem", "Restriktion", "Stenose", "Astma"]

def build_model(dls1, dls2, optimizer, dropout, num_classes, learning_rate):

	model = Sequential()
	model.add(Dense(dls1, input_dim=78, activation='relu'))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	if optimizer=="sgd":
		opt = optimizers.SGD(lr=learning_rate)
	else:
		opt = optimizers.Adam(lr=learning_rate)
	model.compile(loss='categorical_crossentropy',
					optimizer=opt,
					metrics=['accuracy'])

	return model

def log_model_params(model, wandb_config, args, length):
	""" Extract params of interest about the model (e.g. number of different layer types).
		Log these and any experiment-level settings to wandb """

	wandb_config.update({
		"epochs" : args.epochs,
		"batch_size" : args.batch_size,
		"num_classes" : args.num_classes,
		"optimizer" : args.optimizer,
		"dropout" : args.dropout,
		"learning_rate": args.learning_rate,
		"Train_data_size": length
	})

def run_experiment(args):
	""" Build model and data generators; run training"""
	wandb.init(project="KNN_v3_softmax_sweep_1layer_size_change")

	model = build_model(args.dense_layer1, args.dense_layer2, args.optimizer, args.dropout, args.num_classes, args.learning_rate)
	
	div = args.train_size_div

	#import data
	a_file = open("input_data_versch_train.pkl", "rb")
	X_train = pickle.load(a_file)
	a_file.close()

	a_file = open("input_data_versch_test.pkl", "rb")
	X_test = pickle.load(a_file)
	a_file.close()

	a_file = open("output_data_versch_train.pkl", "rb")
	y_train = pickle.load(a_file)
	a_file.close()
	a_file = open("output_data_versch_test.pkl", "rb")
	y_test = pickle.load(a_file)
	a_file.close()

	X_train_2 = []
	y_train_2 = []
	X_test_2 = []
	y_test_2 = []

	a = 0
	b = 0

	while(a < len(X_train)):
		for c in range(0, 75*4):
			X_train_2.append(X_train[a])
			y_train_2.append(y_train[a])
			a += 1
		for c in range(0, 75*4*(div-1)):
			a += 1

	# while(b < len(X_test)):
	# 	for c in range(0, 75*4):
	# 		X_test_2.append(X_test[b])
	# 		y_test_2.append(y_test[b])
	# 		b += 1
	# 	for c in range(0, 75*4*(div-1)):
	# 		b += 1


	X_train = numpy.array(X_train_2)
	X_test = numpy.array(X_test)
	y_train = numpy.array(y_train_2)
	y_test = numpy.array(y_test)

	# log all values of interest to wandb
	log_model_params(model, wandb.config, args, len(X_train))

	# core training method  
	model.fit(X_train, y_train,  validation_data=(X_test, y_test), epochs=args.epochs,
    callbacks=[WandbCallback(data_type="values", labels=labels)])

	# save the model weights
	save_model_filename = args.model_name + ".h5"
	model.save(save_model_filename)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	# args
	#----------------------------
	parser.add_argument(
		"-m",
		"--model_name",
		type=str,
		default="v3_softmax_sweep_onelayer_size_change",
		help="Name of this model/run (model will be saved to this file)")
	parser.add_argument(
		"-b",
		"--batch_size",
		type=int,
		default=75,	#Extended because of new datastructure
		help="Batch size")
	parser.add_argument(
		"-ts",
		"--train_size_div",
		type=int,
		default=2,
		help="Train size Division")
	parser.add_argument(
		"-c",
		"--num_classes",
		type=int,
		default=5,
		help="Number of classes to predict")
	parser.add_argument(
		"-d",
		"--dropout",
		type=float,
		default=0.0,
		help="Dropout before the last fc layer") 
	parser.add_argument(
		"-e",
		"--epochs",
		type=int,
		default=10,
		help="Number of training epochs")
	parser.add_argument(
		"-o",
		"--optimizer",
		type=str,
		default="adam",
		help="Learning optimizer")
	parser.add_argument(
		"-l",
		"--learning_rate",
		type=float,
		default=0.001,
		help="Learning rate")
	parser.add_argument(
		"-dl1",
		"--dense_layer1",
		type=int,
		default=30,
		help="Dense Layer size 1") 
	parser.add_argument(
		"-dl2",
		"--dense_layer2",
		type=int,
		default=10,
		help="Dense Layer size 2") 
	parser.add_argument(
		"-q",
		"--dry_run",
		action="store_true",
		help="Dry run (if set, do not log to wandb)")
	
	args = parser.parse_args()

	# easier iteration/testing--don't log to wandb if dry run is set
	if args.dry_run:
		os.environ['WANDB_MODE'] = 'dryrun'
	
	run_experiment(args) 

