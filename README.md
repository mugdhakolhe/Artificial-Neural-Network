# Artificial-Neural-Network

Configurable parameters in code:
•	No of hidden layers: User can enter the no of hidden layers in the neural network 
•	No of units in each hidden layer: After entering the no of hidden layers the user wants in the neural network, the user will be prompted to enter no of hidden units in each hidden layer
•	Activation function: The user has a choose to choose amongst ReLu, Sigmoidal or tanh as activation function for each hidden layer. He will be asked to enter a function for each hidden layer (Press 1 for Sigmoidal, 2 for tanh and 3 for ReLu). The default is set to tanh in case user enters wrong no. 
•	Type of architecture: The user can either choose a fully connected network or can self- design the connections between the neurons. To do so the user will be shown the neurons one by one along with what choices he/she has to connect that neuron with in the next hidden layer. User has to press 1 to have connection between the shown neurons or 0 for no connection.
To stop overfitting user can chose parameters like:

•	Add momentum to converge weights quickly: On pressing 1 (i.e. yes) the suer can enter the threshold value by with the weights will converge.
•	User can choose to add weight decay: On pressing yes (i.e. 1) user can enter the factor by which each of the weight updates will show down to avoid overfitting.

User can choose between:
1.	Condition on no. of iterations: Enter fixed no of epochs or choose a threshold such that when error reaches a threshold we can stop the training and use those weights for testing. 
2.	Cross-validation: The training-data will be broken down into two parts by some random no r from 0 to 1 which will be the ratio. According to value of r the data will be sent for training and validation and error will be calculated on validation set. The set of best weights will be maintained. If the difference between training weights currently used give error on validation set and the best weights tested so far on validation sets is greater then a threshold we stop the training.
3.	K-fold Cross-Validation: We repeat the above process k no of times as entered by the user. Each time splitting training data into different ratios. Then we calculation the no of iterations in each time we perform cross validation and then take the average of them. Then we use that no of iterations to train the entire training set and   the used those weights obtained for testing.     

Also, user can change values of parameters such as :   

•	The learning rate denoted by ‘alpha’
•	No of outputs in output units in output layer of neural network
•	The activation function of output layer
