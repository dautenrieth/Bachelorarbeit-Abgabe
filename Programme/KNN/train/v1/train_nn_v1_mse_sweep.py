# train_nn_v1_mse_sweep.py
# Trainingsprogramm
# 2 Hidden-Layers, Relu, Optimierervariation
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

from sklearn.model_selection import train_test_split

labels =["Normal", "Empysem", "Restriktion", "Stenose", "Astma"]

def build_model(optimizer, dropout, num_classes, learning_rate):

	model = Sequential()
	model.add(Dense(50, input_dim=78, activation='relu'))
	model.add(Dense(10))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))
	model.add(Dense(num_classes))
	model.add(Activation('relu'))

	if optimizer=="sgd":
		opt = optimizers.SGD(lr=learning_rate)
	else:
		opt = optimizers.Adam(lr=learning_rate)
	model.compile(loss='mean_squared_error',
					optimizer=opt,
					metrics=['accuracy'])

	return model

def log_model_params(model, wandb_config, args):
	""" Extract params of interest about the model (e.g. number of different layer types).
		Log these and any experiment-level settings to wandb """

	wandb_config.update({
		"epochs" : args.epochs,
		"batch_size" : args.batch_size,
		"num_classes" : args.num_classes,
		"optimizer" : args.optimizer,
		"dropout" : args.dropout,
		"learning_rate": args.learning_rate
	})

def run_experiment(args):
	""" Build model and data generators; run training"""
	wandb.init(project="KNN_v1_mse_sweep")

	model = build_model(args.optimizer, args.dropout, args.num_classes, args.learning_rate)
	# log all values of interest to wandb
	log_model_params(model, wandb.config, args)

	#import data
	a_file = open("input_data.pkl", "rb")
	X = pickle.load(a_file)
	a_file.close()

	a_file = open("output_data.pkl", "rb")
	y = pickle.load(a_file)
	a_file.close()

	X = numpy.array(X)
	y = numpy.array(y)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


	# core training method  
	model.fit(X_train, y_train,  validation_data=(X_test, y_test), epochs=args.epochs,
    callbacks=[WandbCallback(data_type="values", labels=labels)])

	# save the model weights
	save_model_filename = args.model_name + ".h5"
	model.save(save_model_filename)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	# Strongly recommended args
	#----------------------------
	parser.add_argument(
		"-m",
		"--model_name",
		type=str,
		default="v0.1_mse",
		help="Name of this model/run (model will be saved to this file)")
	
	# Optional args
	#----------------------------
	parser.add_argument(
		"-b",
		"--batch_size",
		type=int,
		default=32,
		help="Batch size")
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
		default=0.3,
		help="Dropout before the last fc layer") 
	parser.add_argument(
		"-e",
		"--epochs",
		type=int,
		default=50,
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
		default=0.01,
		help="Learning rate") 
	parser.add_argument(
		"-q",
		"--dry_run",
		action="store_true",
		help="Dry run (if set, do not log to wandb)")
	
	args = parser.parse_args()

	# easier iteration/testing--don't log to wandb if dry run is set
	if args.dry_run:
		os.environ['WANDB_MODE'] = 'dryrun'


	# if args.notes
	#   os.environ['WANDB_NOTES'] = args.notes
	
	run_experiment(args) 

