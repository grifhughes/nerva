# Nerva

Nerva is a simple 2-layer nerual network implementation for Intel CPUs written in C.  It utilizes the Intel MKL's CBLAS routines to compute the forward and 
backward passes.  Currently, the net uses the relu activation function + a softmax output layer, along with vanilla SGD and L2 regularization for training.  
The training data is also Gaussian normalized.  

# Performance

I tested the MNIST dataset with a 784-200-10 network with the parameters shown in main.c, obtaining an error rate of 1.49% averaged over 10 runs, with an 
average training time of 3m10s on my i5-6440HQ. 

# In the future...

I plan on adding support for more than 1 hidden layer, as well as testing dropout / momentum based SGD.  I might look at adding minibatch SGD, although I like 
the ability to learn online with vanilla.
