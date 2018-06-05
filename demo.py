#!/usr/bin/python3
# Demo for the InfoSec2 (pro)seminar
# Topic: Robustness of Neural Networks
#
# June 2018
# Maximilian Mayerl, Björn Meusburger

# Notes:
# This demo uses keras with theano.
# To make sure this code works, the keras backend must be configures to be theano
# and image_data_format needs to be set to channels_first.

import argparse
import pickle
import numpy as np
import theano
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist


# *********************** Parse CLI arguments ***********************
parser = argparse.ArgumentParser(description="Demo script for the InfoSec2 proseminar of 2018 abount rebustness of neural networks.")
parser.add_argument("-t", "--train", action="store_true", help="Train neural network models on MNIST training data and save them to disk.");
parser.add_argument("-e", "--evaluate", action="store_true", help="Evaluate the given neural network models on untampered MNIST test data.");
parser.add_argument("-d1", "--demo_single_pixel", action="store_true", help="Demo the single pixel perturbations.");
# ToDo: Add other perturbation demo options

parser.add_argument("-o1", "--out_cnn1", default="cnn1.pickle", help="Output path for the first CNN model.");
parser.add_argument("-o2", "--out_cnn2", default="cnn2.pickle", help="Output path for the second CNN model.");

parser.add_argument("-i1", "--in_cnn1", default="cnn1.pickle", help="Input path for the first CNN model.");
parser.add_argument("-i2", "--in_cnn2", default="cnn2.pickle", help="Input path for the second CNN model.");

args = parser.parse_args()

# *********************** Train ***********************
def train_models(x_train, y_train):
    # Build first model (CNN)
    print("Training first CNN model ...")
    model_cnn = Sequential()
    model_cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(1,28,28)))
    model_cnn.add(Conv2D(32, (3, 3), activation="relu"))
    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
    model_cnn.add(Dropout(0.25))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(128, activation="relu"))
    model_cnn.add(Dropout(0.5))
    model_cnn.add(Dense(10, activation="softmax"))
    model_cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model_cnn.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)

    #Build second model (other CNN)
    print("Training second CNN model ...")
    model_cnn2 = Sequential()
    model_cnn2.add(Conv2D(32, (3, 3), activation="tanh", input_shape=(1,28,28)))
    model_cnn2.add(MaxPooling2D(pool_size=(2, 2)))
    model_cnn2.add(Flatten())
    model_cnn2.add(Dense(128, activation="tanh"))
    model_cnn2.add(Dense(10, activation="softmax"))
    model_cnn2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model_cnn2.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)

    # Save models
    print("Saving models:")
    print("   CNN 1 to ", args.out_cnn1)
    pickle.dump(model_cnn1, open(args.out_cnn1, "wb"))
    print("   CNN 2 to ", args.out_cnn2)
    pickle.dump(model_cnn2, open(args.out_cnn2, "wb"))
    

# *********************** Evaluate ***********************
def evaluate_models(x_test, y_test):
    # Load models
    print("Loading models:")
    print("   CNN 1 from ", args.in_cnn1)
    model_cnn1 = pickle.load(open(args.in_cnn1, "rb"))
    print("   CNN 2 from ", args.in_cnn2)
    model_cnn2 = pickle.load(open(args.in_cnn2, "rb"))

    #Evaluate models
    print("Evaluating CNN 1 ...")
    score_cnn1 = model_cnn1.evaluate(x_test, y_test)
    print("CNN 1 score: ", score_cnn1)

    print("Evaluating CNN 2 ...")    
    score_cnn2 = model_cnn2.evaluate(x_test, y_test)
    print("CNN 2 score: ", score_cnn2)
    

# Import MNIST data as provided by Keras and reshape for Theano backend
# We need this for all actions the script can perform, so we always do this
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
x_train = x_train.astype("float32")
x_train /= 255
x_test = x_test.astype("float32")
x_test /= 255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

#Transfer control according to the CLI arguments given
if args.train:
    train_models(x_train, y_train)

if args.evaluate:
    evaluate_models(x_test, y_test)

#ToDo: Add perturbation demos
