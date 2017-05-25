import os
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt #ploting the process
%matplotlib inline

path = os.getcwd() + '/log.data.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted']) 
#plot so we can see what's the deal:
positive = data[data['Admitted'].isin([1])]  
negative = data[data['Admitted'].isin([0])]
fig, ax = plt.subplots(figsize=(8,6))  
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')  
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')  
ax.legend()  
ax.set_xlabel('Exam 1 Score')  
ax.set_ylabel('Exam 2 Score')  
# sigmoid function - a special case of the logistic function
def sigmoid(z):  
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))

def gradientReg(theta, X, y, learningRate):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])

        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:,i])

    return grad

# add a ones column - this makes the matrix multiplication work out easier
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols]

# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)  
y = np.array(y.values)  
theta = np.zeros(cols-1)  

#check the dimentions:
X.shape, theta.shape, y.shape  

data.describe()
data.plot(kind='scatter', x='Population', y='Profit', figsize=(8,4))  

def computeCost(X, y, theta):  
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

def gradientDescent(X, y, theta, alpha, iters):  
    temp = np.matrix(np.zeros(theta.shape)) #martix to contain temp thetas
    parameters = int(theta.ravel().shape[1]) #number of parameters
    cost = np.zeros(iters) 

    for i in range(iters):
        error = (X * theta.T) - y #Sigma (y'-y)
        for j in range(parameters):
            term = np.multiply(error, X[:,j]) #now we multiple the error in x(i)
            #take last theta and reduce the alpha times 1/examples
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))  #np.sum really sums the sigma

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost

data.insert(0, 'Ones', 1) # append a ones column to the front of the data set
# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  #get all tethas
y = data.iloc[:,cols-1:cols] #get the y

#create a matrix from the dataframe:
X = np.matrix(X.values)  
y = np.matrix(y.values)  
theta = np.matrix(np.array([0,0]))             

# initialize variables for learning rate and iterations
alpha = 0.01  
iters = 1000

# perform gradient descent to "fit" the model parameters
g, cost = gradientDescent(X, y, theta, alpha, iters)  


#plot both the liner model and the error  
x = np.linspace(data.Population.min(), data.Population.max(), 100)  
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(8,4))  
ax.plot(x, f, 'r', label='Prediction')  
ax.scatter(data.Population, data.Profit, label='Traning Data')  
ax.legend(loc=2)  
ax.set_xlabel('Population')  
ax.set_ylabel('Profit')  
ax.set_title('Predicted Profit vs. Population Size')  

fig, ax = plt.subplots(figsize=(8,4))  
ax.plot(np.arange(iters), cost, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch')  


#%reset