# GenNeuAct
This repository contains the code written for my Master's thesis: Exploiting neuron activation values for creating adversarial examples.

### Abstract
The increasing usage of neural networks forms a threat to the cyber security as attacks with adversarial examples can deceive the networks.
Because neural networks can have complex structures with tens of thousands of parameters, they are hard for humans to understand. Hence, existing white-box attacks use very limited network information and most state-of-the-art methods are based on gradient descent. However, intuitively attacks and defenses can be more effective when the user understands the model and uses all information contained by the model.
In this work we further investigate the inner workings of a neural network and consider using intermediate network information for the creation of adversarial examples. We show that neuron activation values can be distinguished by the class of the data point and contain meaningful information about the prediction of the network. Based on this information, we propose a new, gradient-free method for creating adversarial examples based on a genetic algorithm. 
By covering a larger part of the search space and manipulating the neuron activation values, our success rate 
exceeds most state-of-the-art methods, such as DeepFool and RFGSM. We also find that the trade-off between  success rate and distance has a huge impact on the results of a method, wherefore we recommend to carefully balance this trade-off by formulating an optimization formula with a separate loss and distance component. 
