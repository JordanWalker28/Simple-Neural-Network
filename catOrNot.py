'''

@author: Jordan Walker

@Code Sources:
    All sources are harvarded referenced in the report under ther bibliography section
Data loading: Marta, Heriot Watt University
Functions and Model: (SnailDove's blog, 2018)
                     (Look back in respect, 2018)
                     (Shen, 2018)
'''

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage


# Loading the datasets both training and testing
train_dataset = h5py.File('trainCats.h5', "r")
trainSetX = np.array(train_dataset["train_set_x"][:]) # your train set features
trainSetY = np.array(train_dataset["train_set_y"][:]) # your train set labels

test_dataset = h5py.File('testC.h5', "r")
testSetX = np.array(test_dataset["test_set_x"][:]) # your test set features
testSetY = np.array(test_dataset["test_set_y"][:]) # your test set labels


print('X Train: Rows: %d' , trainSetX[0])
print('X Test: Rows: %d' , testSetX[0])

#test case classes i.e. cat / non-cat
classes = np.array(test_dataset["list_classes"][:])


m_train = trainSetX.shape[0] #number of training examples
m_test = testSetX.shape[0] #number of testing examples
num_px = trainSetX.shape[1] #number of pixels per image height x width

#reshape from (num_px,num_px,3) -> single vector (num_px,num_px, *3,1)
trainSetXF= trainSetX.reshape(trainSetX.shape[0], -1).T
testSetXF = testSetX.reshape(testSetX.shape[0], -1).T

#standerdize data set by 255
trainSetX = trainSetXF / 255;
testSetX = testSetXF / 255;

#define sigmoid activation function
def sigmoid(z):
    s = 1 / (1 + np.exp(-z));
    return s;

#parameter initialization setting 'w' as a vector of zero
def initialize_with_zeros(dim):
    #used to set the weights and bias for the input network
    w = np.zeros((dim, 1));
    b = 0;
    assert(w.shape == (dim, 1));
    assert(isinstance(b, float) or isinstance(b, int));
    return w, b;

#computes the cost function and the gradient
def propagate(w, b, X, Y):
    
    m = X.shape[1]
    
    #create forward and backward propogations
    #used to find the cost and the gradient
    a = sigmoid(np.dot(w.T, X) + b);
    cost = - 1 / m * (np.dot(Y, np.log(a).T) + np.dot(1 - Y, np.log(1 - a).T));
    dw = 1 / m * np.dot(X,(a - Y).T);
    db = 1 / m * np.sum(a - Y);
    
    assert(dw.shape == w.shape);
    assert(db.dtype == float);
    cost = np.squeeze(cost);
    assert(cost.shape == ());
    
    grads = {"dw": dw,
        "db": db};
    
    return grads, cost;

#Update the parameters using gradient descent
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    
    costs = []
    
    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y);
        dw = grads["dw"];
        db = grads["db"];

        w -= learning_rate * dw;
        b -= learning_rate * db;

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,
            "b": b}
    
    grads = {"dw": dw,
            "db": db}
    
    return params, grads, costs

#function to predict based whether an entrie is labelled as a 0 (non-cat) 1 (cat)
def predict(w, b, X):

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b);
    
    for i in range(A.shape[1]):

        if A[0,i] > 0.5:
            Y_prediction[0,i] = 1;
        else:
            Y_prediction[0,i] = 0;

    assert(Y_prediction.shape == (1, m));

    return Y_prediction;

#created model to use the above created functions in the right order for logistic regression
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    #set the parameters to 0
    w, b = initialize_with_zeros(X_train.shape[0]);
    #Use Gradient Descent
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost);
    #Retreive paramenters
    w = params["w"];
    b = params["b"];
    #predict the test and training examples
    Y_prediction_train = predict(w, b, X_train);
    Y_prediction_test = predict(w, b, X_test);

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    
    d = {"costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train" : Y_prediction_train,
        "w" : w,
        "b" : b,
        "learning_rate" : learning_rate,
        "num_iterations": num_iterations}

    return d;

#Testing of the model
#Test the effects of num_iterations va learning_rate
d = model(trainSetX, trainSetY, testSetX, testSetY, num_iterations = 5000, learning_rate = 0.0001, print_cost = True);

costs = np.squeeze(d['costs']);
plt.plot(costs);
plt.ylabel('cost');
plt.xlabel('iterations (per hundreds)');
plt.title("Learning rate =" + str(d["learning_rate"]));
plt.show();

#Test different learning rates against one another
learning_rates = [0.01, 0.001, 0.0001];
models = {};
for i in learning_rates:
    print ("learning rate is: " + str(i));
    models[str(i)] = model(trainSetX, trainSetY, testSetX, testSetY, num_iterations = 5000, learning_rate = i, print_cost = False);
    print ('\n');

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]));

plt.ylabel('cost');
plt.xlabel('iterations(hundred)');

legend = plt.legend(loc='upper center', shadow=True);
plt.show();


