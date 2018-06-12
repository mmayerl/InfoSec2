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
import random
import math
import numpy as np
from scipy.optimize import minimize, Bounds
import theano
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from scipy.misc import imsave
from keras import backend as K
import matplotlib.pyplot as plt
from foolbox.models import TheanoModel, KerasModel
from foolbox.criteria import TargetClassProbability
from foolbox.attacks import LBFGSAttack
from foolbox.adversarial import Adversarial


# *********************** Parse CLI arguments ***********************
parser = argparse.ArgumentParser(description="Demo script for the InfoSec2 proseminar of 2018 abount rebustness of neural networks.")
parser.add_argument("-t", "--train", action="store_true", help="Train neural network models on MNIST training data and save them to disk.");
parser.add_argument("-e", "--evaluate", action="store_true", help="Evaluate the given neural network models on untampered MNIST test data.");
parser.add_argument("-d1", "--demo_single_pixel", action="store_true", help="Demo the single pixel perturbations.");
parser.add_argument("-d2", "--demo_all_pixel", action="store_true", help="Demo the constant all pixel perturbations.");
parser.add_argument("-d3", "--demo_gaussian", action="store_true", help="Demo the gaussian perturbations.");
parser.add_argument("-d4", "--demo_universal", action="store_true", help="Demo universal perturbations.");
parser.add_argument("-d5", "--demo_lbfgs", action="store_true", help="Demo LBFGS attack (Szegedy et al. 2013).")
parser.add_argument("-d6", "--demo_adversarial", action="store_true", help="Demo simple adversarial attack (Berkeley Tutorial).")

# ToDo: Add other perturbation demo options

parser.add_argument("-n", "--num_samples", default=100, help="The number of samples that we attempt to fool.");

parser.add_argument("-o1", "--out_cnn1", default="cnn1.hdf5", help="Output path for the first CNN model.");
parser.add_argument("-o2", "--out_cnn2", default="cnn2.hdf5", help="Output path for the second CNN model.");

parser.add_argument("-i1", "--in_cnn1", default="cnn1.hdf5", help="Input path for the first CNN model.");
parser.add_argument("-i2", "--in_cnn2", default="cnn2.hdf5", help="Input path for the second CNN model.");

parser.add_argument("-if", "--image_folder", default="images/", help="Output folder for the generated images.");

args = parser.parse_args()

# *********************** Train ***********************
def train_models(x_train, y_train):
    # Build first model (CNN)
    # Source: ########
    print("Training first CNN model ...")
    model_cnn1 = Sequential()
    model_cnn1.add(Conv2D(32, (3, 3), activation="relu", input_shape=(1,28,28)))
    model_cnn1.add(Conv2D(32, (3, 3), activation="relu"))
    model_cnn1.add(MaxPooling2D(pool_size=(2, 2)))
    model_cnn1.add(Dropout(0.25))
    model_cnn1.add(Flatten())
    model_cnn1.add(Dense(128, activation="relu"))
    model_cnn1.add(Dropout(0.5))
    model_cnn1.add(Dense(10, activation="softmax"))
    model_cnn1.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model_cnn1.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)

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
    model_cnn1.save(args.out_cnn1)
    print("   CNN 2 to ", args.out_cnn2)
    model_cnn2.save(args.out_cnn2)
    

# *********************** Evaluate ***********************
def evaluate_models(x_test, y_test):
    # Load models
    print("Loading models:")
    print("   CNN 1 from ", args.in_cnn1)
    model_cnn1 = load_model(args.in_cnn1)
    print("   CNN 2 from ", args.in_cnn2)
    model_cnn2 = load_model(args.in_cnn2) 

    #Evaluate models
    print("Evaluating CNN 1 ...")
    score_cnn1 = model_cnn1.evaluate(x_test, y_test)
    print("Metrics: ", model_cnn1.metrics_names)
    print("CNN 1 score: ", score_cnn1)

    print("Evaluating CNN 2 ...")    
    score_cnn2 = model_cnn2.evaluate(x_test, y_test)
    print("Metrics: ", model_cnn2.metrics_names)
    print("CNN 2 score: ", score_cnn2)
    

# *********************** Perturbation helpers ***********************
def save_image(path, sample):
    img = np.reshape(sample, (28, 28))
    imsave(path, img)


def get_n_correct(x_test, y_test, model, n):
    samples = []
    for x in zip(x_test, y_test):
        img = np.expand_dims(x[0], axis=0)
        res = model.predict_classes(img)
        if x[1][res[0]] == 1:
            samples.append(img)

        if len(samples) == n:
            break

    return samples


# Plot original image, adversarial example and perturbation
# Taken from the foolbox tutorial: https://foolbox.readthedocs.io/en/latest/user/tutorial.html
def plot_adversarial_example(image, adversarial):
    plt.subplot(1, 3, 1)
    plt.imshow(image)

    plt.subplot(1, 3, 2)
    plt.imshow(adversarial)

    plt.subplot(1, 3, 3)
    plt.imshow(adversarial - image)

    plt.show()


def demo_perturbation(x_test, y_test, perturb_fn, data):
    # Step 1: Load models
    print("Loading models ...")
    K.set_learning_phase(0) # set phase to testing phase - needed for foolbox
    models = [ load_model(args.in_cnn1), load_model(args.in_cnn2) ]

    for model in models:
        successful_perturbs = 0.0
        # Step 2: Get random correctly classified samples
        print("Drawing samples for correctly classified instances ...")
        samples = get_n_correct(x_test, y_test, model, args.num_samples)
        adversarial = samples[:]

        # Step 3: Try to perturb a single pixel in every sample to fool the model
        print("Attempting to perturb those samples ...")
        for i in range(len(samples)):
            print("   Sample ", i)
            success, perturbed_sample = perturb_fn(samples[i], model, data)
            successful_perturbs += success

            # Save images
            if success == 1:
                save_image(args.image_folder + str(i) + ".png", samples[i])
                save_image(args.image_folder + str(i) + "_perturbed.png", perturbed_sample)

            adversarial[i] = perturbed_sample

            # Plot first successful example
            if success == 1 and successful_perturbs == 1:
                plot_adversarial_example(samples[i][0,0,:,:], adversarial[i][0,0,:,:])

        # Step 4: Calculate the fooling rate
        fool_rate = successful_perturbs / args.num_samples
        print("Achieved fooling rate: ", fool_rate)


# *********************** Single pixel perturbations ***********************
def perturb_single_pixel(sample, model, data):
    # We try to perturb every pixel in the given image. 
    # Once we find a pixel whose value we can change to
    # misclassify the image, we are done.
    # the perturbation we try is either setting 
    # the given pixel to 1 or to 0.

    correct_class = model.predict_classes(sample)

    for x in range(28):
        for y in range(28):
            img = np.copy(sample)

            # Set to 1 - missclassify?
            if sample[0, 0, y][x] != 1.0:
                img[0, 0, y][x] = 1.0
                pred_class =  model.predict_classes(img)
                if pred_class != correct_class:
                    return (1, img)

            # Set to 0 - missclassify?
            if sample[0, 0, y][x] != 0.0:
                img[0, 0, y][x] = 0.0
                pred_class =  model.predict_classes(img)
                if pred_class != correct_class:
                    return (1, img)

    #We did't manage to find a perturbation
    return (0, sample)


def demo_single_pixel(x_test, y_test):
    demo_perturbation(x_test, y_test, perturb_single_pixel, None)
 

# *********************** All pixel perturbations ***********************
def perturb_all_pixels(sample, model, data):
    # We try to perturb all pixels in the given image. 
    # We do this by adding a constant value to all pixels,
    # clipped to [0, 1].

    correct_class = model.predict_classes(sample)

    for i in range(1, 25):
        offset = i / 100.0

        # Negative adjustment
        img = np.copy(sample)
        img -= offset
        img = np.clip(img, 0.0, 1.0)

        pred_class =  model.predict_classes(img)
        if pred_class != correct_class:
            return (1, img)

        # Positive adjustment
        img = np.copy(sample)
        img += offset
        img = np.clip(img, 0.0, 1.0)

        pred_class =  model.predict_classes(img)
        if pred_class != correct_class:
            return (1, img)

    return (0, sample)


def demo_all_pixel(x_test, y_test):
    demo_perturbation(x_test, y_test, perturb_all_pixels, None)


# *********************** Gaussian perturbations ***********************
def perturb_gaussian(sample, model, data):
    # We try to perturb all pixels in the given image with a 
    # Gaussian perturbation.

    correct_class = model.predict_classes(sample)

    img = np.copy(sample)
    img += data

    pred_class =  model.predict_classes(img)
    if pred_class != correct_class:
        return (1, img)

    return (0, sample)


def demo_gaussian(x_test, y_test):
    gaussian = np.random.normal(0, 0.1, size=(1, 1, 28, 28))
    demo_perturbation(x_test, y_test, perturb_gaussian, gaussian)


# *********************** LBFGS perturbations ***********************
def perturb_lbfgs(sample, model, data):
    # Perturb images using LBFGS attack by Szegedy et al. using the foolbox library
    # Based on the tutorial: https://foolbox.readthedocs.io/en/latest/user/tutorial.html

    # create model for foolbox
    foolbox_model = KerasModel(model, (0.0, 1.0), channel_axis=1)
    #foolbox_model = TheanoModel(model.input, model.layers[-2].output, (0.0, 1.0), 10, channel_axis=1)

    # get correct class
    correct_class = model.predict_classes(sample)

    # set target to be next higher class (and 0 for 9)
    target_class = (correct_class+1)%10

    # set attack criterion to be 90% target class probability
    criterion = TargetClassProbability(target_class, p=0.90)

    # create attack on model with given criterion
    attack = LBFGSAttack()

    #print(sample[0,:,:,:].shape)

    # generate adversarial example
    # sample needs to be transformed from (batchsize, channels, rows, cols) format to (height, width, channels) for
    # foolbox, but that leads to problems with the model
    transformed_sample = sample[0,:,:,:] # remove batch size
    transformed_sample = np.swapaxes(transformed_sample, 0, 1) # swap channels to axis 1
    transformed_sample = np.swapaxes(transformed_sample, 1, 2) # swap channels to axis 2
    # print(sample[0,:,:,:])
    # print(transformed_sample)
    ad_ins = Adversarial(foolbox_model, criterion, transformed_sample, correct_class)

    adversarial = attack(ad_ins)

    # get class of adversarial example
    pred_class = model.predict_classes(adversarial)
    if pred_class != correct_class:
        return (1, adversarial)

    return (0, sample)


def demo_lbfgs(x_test, y_test):
    demo_perturbation(x_test, y_test, perturb_lbfgs, None)


# *********************** Old Adversarial perturbations ***********************
def adv_cost_function_old(perturbation, sample, y_goal, model):
    # wrongly implemented cost function
    # reshape input matrix since scipy flattens it
    perturbation = perturbation.reshape((1,1,28,28))

    # try to minimize the distance between the goal class and the class of the sample plus perturbation
    return (1/2)*(y_goal**2)-(model.predict_classes(sample + (perturbation-0.5))**2)


def perturb_adversarial_old(sample, model, data):
    # Perturb images using kind of an optimized gaussian? this was a failed implementation
    # get correct class
    correct_class = model.predict_classes(sample)

    # set target to be next higher class (and 0 for 9)
    target_class = (correct_class+5)%10

    # minimize cost
    start = np.random.normal(.5, .25, (1,1,28,28))
    minimized = minimize(adv_cost_function, start, args=(sample, target_class, model), method='Nelder-Mead', options={'disp': True})

    adversarial = sample + minimized.x.reshape((1,1,28,28))

    # use this to plot all generated examples (debugging)
    # plot_adversarial_example(sample[0,0,:,:], adversarial[0,0,:,:])

    # get class of adversarial example
    pred_class = model.predict_classes(adversarial)

    # print original and detected class
    print("{} -> {}".format(correct_class, pred_class))

    if pred_class != correct_class:
        return (1, adversarial)

    return (0, sample)


def demo_adversarial_old(x_test, y_test):
    demo_perturbation(x_test, y_test, perturb_adversarial_old, None)


# *********************** Real Adversarial perturbations ***********************
def adv_cost_function(start, sample, y_goal, model):
    # cost function from https://ml.berkeley.edu/blog/2018/01/10/adversarial-examples/
    # reshape input matrix since scipy flattens it
    start = start.reshape((1,1,28,28))

    goal = np.zeros((10, 1))
    goal[y_goal] = 1

    # try to minimize: output should be classified as goal class but be close to sample class
    return (1/2)*((np.linalg.norm(goal - model.predict(start)))**2) + .05 * (np.linalg.norm(start - sample)**2)


def perturb_adversarial(sample, model, data):
    # Perturb images using an attack based on the description at https://ml.berkeley.edu/blog/2018/01/10/adversarial-examples/
    # get correct class
    correct_class = model.predict_classes(sample)

    # set target to be next higher class (and 0 for 9)
    target_class = (correct_class+5)%10

    # minimize cost
    start = np.random.normal(.5, .3, (1,1,28,28))
    minimized = minimize(adv_cost_function, start, args=(sample, target_class, model), method='BFGS', options={'disp': True})

    adversarial = minimized.x.reshape((1,1,28,28))

    # use this to plot all generated examples (debugging)
    # plot_adversarial_example(sample[0,0,:,:], adversarial[0,0,:,:])

    # get class of adversarial example
    pred_class = model.predict_classes(adversarial)

    # print original and detected class
    print("{} -> {}".format(correct_class, pred_class))
    
    if pred_class != correct_class:
        return (1, adversarial)

    return (0, sample)


def demo_adversarial(x_test, y_test):
    demo_perturbation(x_test, y_test, perturb_adversarial, None)


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

if args.demo_single_pixel:
    demo_single_pixel(x_test, y_test)

if args.demo_all_pixel:
    demo_all_pixel(x_test, y_test)

if args.demo_gaussian:
    demo_gaussian(x_test, y_test)

if args.demo_lbfgs:
    demo_lbfgs(x_test, y_test)

if args.demo_adversarial:
    demo_adversarial(x_test, y_test)
