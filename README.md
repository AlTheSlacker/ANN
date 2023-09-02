# ANN
 Artificial Neural Networks for Unity
 
 This is a cleaned up version of https://github.com/AlTheSlacker/NeuralNet_Basic and includes a visualiser, which it can function without, but helps understand the training process better.
 
 There are some significant improvements: 
 
 1) Input and output fields are automatically normalised based on training data - this makes setting up a useful network much easier.
 2) The visualiser allows you to see how the network is functioning and look out for saturated layers.
 3) Added the tanh activation function.
 4) Added the ability to train the network with a target error level.
 
 I would really like to move the back-propagation part to a compute shader, but I just can't figure out how to convert all the object data to appropriate 1D arrays and wrap some code around it. I'd love to hear from anyone who figures it out and does not mind sharing.

 Warning! Things have moved on a lot since I first wrote this and I would encourage anyone interested in ANNs to follow this excellent lecture series and then write something much better than my efforts here!
 
https://www.youtube.com/watch?v=VMj-3S1tku0
https://www.youtube.com/watch?v=PaCmpygFfXo
https://www.youtube.com/watch?v=TCH_1BHY58I
https://www.youtube.com/watch?v=P6sfmUTpUmc
https://www.youtube.com/watch?v=q8SA3rM6ckI
https://www.youtube.com/watch?v=t3YJ5hKiMQ0
