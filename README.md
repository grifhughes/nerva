# Nerva

DISCLAIMER - main.c is only to show how the library functions would work in a real program, if you wish to run it yourself you need to download mnist_train.csv 
and mnist_test.csv online ([here](https://pjreddie.com/projects/mnist-in-csv/) is where I got mine) (and also have the MKL installed).
 
Nerva is a simple 2-layer nerual network implementation for Intel CPUs written in C.  It utilizes the Intel MKL's CBLAS routines to compute the forward and
backward passes.  Currently, the net uses the relu activation function + a softmax output layer, along with vanilla SGD and L2 regularization for training.
The training data is also automatically Gaussian normalized.  

# IO

The IO functions expect the input data files in .csv, with the class of the example as the first number in each list.  

# Performance

I tested the MNIST dataset with a 784-200-10 network with the hyperparameters shown in main.c, obtaining an error rate of 1.49% averaged over 10 runs, with an 
average training time of 3m10s (1,800,000 iterations, 30 epochs) on my i5-6440HQ. 

# In the future...

I plan on adding support for more than 1 hidden layer as well as testing dropout / momentum based SGD.  I might look at adding minibatch SGD, although I like 
the ability to learn online with vanilla. Also, possibly implementing a parallel scheme in which 4 different models are trained on the same data, and the 
gradients averaged to give the final updates. CUDA extensions are in the works.  
