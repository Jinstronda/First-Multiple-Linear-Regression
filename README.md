![image](https://github.com/user-attachments/assets/7cfc3469-db94-4b27-850d-aa1b0603a929)Studying ML as a side hobbie, i want to master it in 3-6 months (first ML then DL), i am very beginner but i am already falling in love with it. Love the applications of math and statistics on it

For Multiple Linear Regression 
# Functions:
calcular_hypothesis(x_train,w_vector,b) - Calculate the current hypothesis of the model using dot vector multiplication

calcular_custo_e_vetorgradiente(x_test,y_test,w_vector,b) - Calculate the Cost and the Vector Gradient for each cost, it uses vector dot multiplication to do that, comparing the real prices with the predicted prices, after that it calculate the derivative for w and b.

calcular_custo(x_train,y_test, w_vector, b) - Calculate cost so we can later plot it against iterations

logfunctioncost(iterations,cost) - Graph the Function cost with a log scale so we can see how it decreases with each iteration

def normalization(x_train) - Normalize a set of data using Z-SCORE, calculating the mean of it and the standard deviation, then it returns the new MATRIX plus these values so we can use them later to normalize new data

def new_values_normalization(mean,std, house) - Normalize new data that we want to predict 

# Variables:
x_train0 = The Matrix with the input values used to train the Model (before normalization)
y_train = Vector with output values used to train the model
m = Number of training examples
n = Number of Features
l = Learning rate of the model
b = Bias
w_vector = Weight Vectors, one weight for each n
cost = Save the cost 
iterations = Save the iterations so we can relate them to the cost

# Formulas used 
![image](https://github.com/user-attachments/assets/7d8d0285-8851-4e46-9beb-2342613a1c3d)
![image](https://github.com/user-attachments/assets/7331c299-fa74-4d96-a0fe-a5787cf8b1c0) 

