    For my experimentation I ran several trials, altering one variable at a time (or 2 if the variables were closely related like # of hidden layers and nodes per hidden layer). 
    I found that increasing the epoch past the original 10 would not necessarily improve the accuracy actually may lead to lower accuracy. 
    I added more convolutional and pooling layers on trial 5 which led to a slightly higher accuracy than the original 1 layer each. 
    Removing dropout lowered the accuracy while increasing the dropout from 0.5 to 0.85 caused a major drop in accuracy.
    Increasing the kernel and pooling size caused only a very slight decrease in accuracy.
    Adding a hidden layer using tanh activation instead of relu caused a major drop in accuracy.
    Adding more hidden layers, each with less nodes but so that the total number of hidden nodes is the same caused a small dip in accuracy
    More convolutional filters caused a small increase in accuracy

    For the final model, I:
        increased convolutional filters from 32 to 64
        increased convolutional layers from 1 to 3
        increased pooling layers from 1 to 3
        increased hidden layers form 1 to 2
        increased nodes per hidden layer from 128 to 256


TRIAL : 1
PARAMETERS:
    Epochs = 10
    Convolutional layers = 1
    Convolutional parameters = 32 filters, 3x3 kernel, relu activation
    Pooling layers = 1
    Pooling size = 2x2
    Hidden layers = 2
    Hidden layer parameters = 256 nodes, relu activation
    Dropout = 0.5
    Output Layer parameter = softmax activation
ACCURACY : 0.9282

TRIAL : 2
PARAMETERS:
    Epochs = 10
    Convolutional layers = 1
    Convolutional parameters = 64 filters, 3x3 kernel, relu activation
    Pooling layers = 1
    Pooling size = 2x2
    Hidden layers = 2
    Hidden layer parameters = 256 nodes, relu activation
    Dropout = 0.5
    Output Layer parameter = softmax activation
ACCURACY : 0.9440

TRIAL : 3
PARAMETERS:
    Epochs = 10
    Convolutional layers = 1
    Convolutional parameters = 32 filters, 3x3 kernel, relu activation
    Pooling layers = 1
    Pooling size = 2x2
    Hidden layers = 2
    Hidden layer parameters = one with 256 nodes, relu activation and one with 512 nodes, tanh activation
    Dropout = 0.5
    Output Layer parameter = softmax activation
ACCURACY : 0.0562

TRIAL : 4
PARAMETERS:
    Epochs = 15
    Convolutional layers = 1
    Convolutional parameters = 32 filters, 3x3 kernel, relu activation
    Pooling layers = 1
    Pooling size = 2x2
    Hidden layers = 2
    Hidden layer parameters = 256 nodes, relu activation
    Dropout = 0.5
    Output Layer parameter = softmax activation
ACCURACY : 0.9323

TRIAL : 5
PARAMETERS:
    Epochs = 10
    Convolutional layers = 3
    Convolutional parameters = 32 filters, 3x3 kernel, relu activation
    Pooling layers = 3
    Pooling size = 2x2
    Hidden layers = 2
    Hidden layer parameters = 256 nodes, relu activation
    Dropout = 0.5
    Output Layer parameter = softmax activation
ACCURACY : 0.9535

TRIAL : 6
PARAMETERS:
    Epochs = 10
    Convolutional layers = 1
    Convolutional parameters = 32 filters, 5x5 kernel, relu activation
    Pooling layers = 1
    Pooling size = 5x5
    Hidden layers = 2
    Hidden layer parameters = 256 nodes, relu activation
    Dropout = 0.5
    Output Layer parameter = softmax activation
ACCURACY : 0.9230

TRIAL : 7
PARAMETERS:
    Epochs = 10
    Convolutional layers = 1
    Convolutional parameters = 32 filters, 3x3 kernel, relu activation
    Pooling layers = 1
    Pooling size = 2x2
    Hidden layers = 2
    Hidden layer parameters = 256 nodes, relu activation
    Dropout = 0
    Output Layer parameter = softmax activation
ACCURACY : 0.9061

TRIAL : 8
PARAMETERS:
    Epochs = 10
    Convolutional layers = 1
    Convolutional parameters = 32 filters, 3x3 kernel, relu activation
    Pooling layers = 1
    Pooling size = 2x2
    Hidden layers = 2
    Hidden layer parameters = 256 nodes, relu activation
    Dropout = 0.85
    Output Layer parameter = softmax activation
ACCURACY : 0.0551

TRIAL : 9
PARAMETERS:
    Epochs = 10
    Convolutional layers = 1
    Convolutional parameters = 32 filters, 3x3 kernel, relu activation
    Pooling layers = 1
    Pooling size = 2x2
    Hidden layers = 5
    Hidden layer parameters = 64 nodes, relu activation
    Dropout = 0.5
    Output Layer parameter = softmax activation
ACCURACY : 0.8743

TRIAL : 10
PARAMETERS:
    Epochs = 10
    Convolutional layers = 1
    Convolutional parameters = 32 filters, 3x3 kernel, relu activation
    Pooling layers = 1
    Pooling size = 2x2
    Hidden layers = 1
    Hidden layer parameters = 1024 nodes, relu activation
    Dropout = 0.5
    Output Layer parameter = softmax activation
ACCURACY : 0.9313