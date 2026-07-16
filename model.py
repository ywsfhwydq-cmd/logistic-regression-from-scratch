# Import the main libraries for the project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the data file
df = pd.read_csv('cell_samples.csv')

# Replace '?' with missing values (NaN)
df['BareNuc'] = df['BareNuc'].replace('?' , np.nan)

# Clean the data: remove empty rows and drop the ID column
df = df.dropna()
df['BareNuc'] = df['BareNuc'].astype(int)
df = df.drop('ID', axis=1)

# Change target classes to 0 and 1
df['Class'] = np.where(df['Class'] ==4 , 1 , 0)

# Split the data into features (X) and target labels (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Reshape y to be a 2D column vector
y = y.reshape(-1, 1)


# Sigmoid function: turns numbers into probabilities between 0 and 1
def sigmoid(x):

    return 1/(1+ np.exp(-x))


# Initialize weights (w) and bias (b) with zeros
def parameters(z):
    w = np.zeros(( z ,1))
    b = 0.0
    
    return w , b


# Forward pass: calculates the model predictions
def forward(x,w,b):
    y= np.dot(x,w)+b
    y_head = sigmoid(y)
    return y_head


# Cost function: calculates the total error of predictions
def cost(y_head,y):

    m = y.shape[0]

    cost = -(1/m) * np.sum(y*np.log(y_head) + (1-y)* np.log(1- y_head))

    return cost


# Backward pass: calculates gradients to fix the error
def Backward(x , y , y_head):

    m = y.shape[0]

    error = y_head - y


    dw = (1/m) *np.dot(x.T,error)
    db = (1/m)* np.sum(error)

    return dw , db


# Set the learning rate and total number of iterations
learning_rate = 0.0002
epoches = 100000

# An empty list to store the cost values during training
cost_history = []

# Update weights and bias to reduce the model error
def update(w , b , dw , db , lr):

    w = w - lr * dw
    b = b - lr * db

    return w , b


# Get the number of features and initialize parameters
num_features = X.shape[1]
w, b = parameters(num_features)


# The main loop to train the model
for i in range(epoches):
   

    # 1. Get current predictions
    y_head = forward(X, w, b)

    # 2. Calculate the current error
    initial_cost = cost(y_head, y)

    # 3. Calculate the gradients
    dw, db = Backward(X, y, y_head)

    # 4. Update parameters and save the cost
    w, b = update(w, b, dw, db, learning_rate)
    cost_history.append(initial_cost)


    
# Print final weights and bias after training
print(w,b)

# Plot a line graph to show how the error drops over time
plt.plot(cost_history , color = 'red' , linewidth=2)
plt.title('How the Error Dropped Over Time (Cost Function)')
plt.xlabel('Number of Epochs (Iterations)')
plt.ylabel('Error Amount (Cost Value)')
plt.grid(True)
plt.show()