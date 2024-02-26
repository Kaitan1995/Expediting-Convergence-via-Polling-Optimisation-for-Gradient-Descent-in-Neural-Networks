# Expediting Convergence via Polling Optimisation for Gradient Descent in Neural Networks Test

An inherent disadvantage remains however where neural networks are deemed "black box" models; their operations depending on sometimes obscure and complex attenuation of weights and biases in interconnected nodes upon introduction of complexities, so as to attempt to learn and make predictions. The amounts of attenuation being tied to an important hyperparameter – the learning rate. As the learning rate affects the ability and efficiency of gradient descent towards the global minima of the cost function after each backpropagation process, having a constant learning rate typically leads to rather disastrous results due to two definitive scenarios (I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. in Adaptive computation and machine learning series. Cambridge, MA: MIT Press, 2017.) 
- When ‘far away’ from the global minima of the cost function, undershoot may occur and results in the model taking longer than practically allowable to converge.
- When ‘near’ the global minima of the cost function, overshoot may occur which results in an excessive time for the model to converge. This is caused by too excessive an update to the weights and biases within the model.

![image](https://github.com/Kaitan1995/Expediting-Convergence-via-Polling-Optimisation-for-Gradient-Descent-in-Neural-Networks/assets/93040738/62c5808e-08d7-4fb2-a1e1-251e8bd96f8f)

To compensate, an ‘adaptive learning rate’ approach is used to mitigate this. A common example of adaptive learning rate is Adaptive Moment Estimation (ADAM) that relies on first (Momentum) and second (Root Mean Square Propagation) raw moment estimates to attune a singular learning rate on every epoch during training. This study suggests a different approach of allowing a diversity of learning rates enabled through a modified ensemble approach within the training phase of neural networks – creating a scenario where multiple base-models are created within each epoch. In the current environment, more validation is deemed necessary before adaptive learning rates become more established and understood (Y. Bengio, “Practical Recommendations for Gradient-Based Training of Deep Architectures,” Neural Networks, Jun. 2012, doi: 10.1007/978-3-642-35289-8_26.) (L. N. Smith, “A Disciplined Approach to Neural Network Hyper-Parameters: Part 1 - Learning Rate, Batch Size, Momentum, and Weight Decay,” ArXiv, vol. abs/1803.09820, 2018, [Online]. Available: https://api.semanticscholar.org/CorpusID:4714223). 

## Polling Method
To facilitate this, an exploratory method dubbed the Polling Method will attempt to utilise an arbitrary number (n) of learning rates (α_1,α_2,…,α_n) of differing reasonable values within a single neural network model to determine a single optimum learning rate at each step of backpropagation. 

1) To begin the Polling Method, a single neural network model is created – With the randomisation of weights and biases, a typical forward pass and backward propagation of errors carried out.
2) The Polling Method adds an additional step before weights and biases updating. Before the actual weights and biases updating, an array of learning rates (α_1,α_2,…,α_n) are introduced and the weights and biases are updated by n instances of the learning rate, creating an array of weights and biases defined by different learning rate values, roughly defined as being different base-models.
3) Out of the n instances of diverged weights and biases (base-models), only a single weights and biases value should be chosen as the final and global weights and biases value to continue the training process. This singular weights and bias value is tied to the best learning rate. To obtain this, for each base-model, their parameters are subjected to forward pass to generate predictions (Different A_2 due to α_1, α_2, …, α_n) against a global training class labels. The learning rate (α_1,α_2,…,α_n) that generates the best prediction (A_2) is deemed the base-model with the highest accuracy.
4) The learning rate (α_1,α_2,…,α_n) associated with this highest base-model accuracy is deemed the singular best learning rate and chosen. The weights and biases associated with this base-model are selected above all other base-models. This system is repeated for as many epochs/iterations in the neural network model. 

## Principle 
The Polling Method is loosely based upon the technique of (voting) ensemble modelling in neural networks, which refers to rather than relying on an output of a single model, multiple models are utilised with varying degrees of cooperation to improve the overall predictive performance of the model through leveraging the diversity of different models, as well as bootstrap aggregating (bagging), a method pioneered by Professor Leo Brieman.

![image](https://github.com/Kaitan1995/Expediting-Convergence-via-Polling-Optimisation-for-Gradient-Descent-in-Neural-Networks/assets/93040738/19a7145e-76e8-4583-a1e9-5fc1cb1d6d75)

In the Bias-Variance Trade-off observed in (1) where E refers to expectation, h_D refers to the classifier using a dataset D, h ̅ refers to the expected classifier D, y refer to the general Ground Truth (correct prediction) and y ̅ refers to the expected value of all models, the composition equates to (in order of) Total Mean Square Error in a model =  Total Variance + Total Bias + Noise. When improving diversity into the dataset (i.e. increasing the number of datasets) as observed in (2), the effect on the minor model can be observed through (3). When this happens. the weak law of large numbers is evoked as observed in (4) as the total variance error as observed in (1) becomes mitigated through transforming h_D (x) towards h ̅(x).

The Polling Method suggests utilising the bagging approach with a few marked differences: 	
- Instead of having multiple subsets (typically with replacement) of the same dataset in conventional bagging, the entire training dataset will be utilised for each subset. This is to account for MNIST’s low number of examples (<100,000 examples).
- Instead of having (typically) the same models for each subset of the same dataset in conventional bagging, differing learning rates will be utilised in each model (base-model) as a means to introduce diversity.
- Instead of having an aggregate (mean) ensemble model voting approach for the multiple subsets of the same dataset in conventional bagging, the sub-model (bag) that outputs the highest accuracy will be utilised in the Polling Method (different sub-models/base-models will produce different accuracy ratings due to having different learning rates).

## Artificial Neural Network Model

A computer vision-abled ANN will be trained with the Modified National Institute of Standards and Technology (MNIST) dataset for all labels (42,000 examples). 

![image](https://github.com/Kaitan1995/Expediting-Convergence-via-Polling-Optimisation-for-Gradient-Descent-in-Neural-Networks/assets/93040738/3a95fd41-5767-4b6a-bf35-01b6325a73e7)

## Convolutional Neural Network Model

A computer vision-abled CNN will be trained with the MNIST dataset for all labels (42,000 examples). In this experiment, the CNN is based upon the Numpy-CNN code designed by Alescontrela that is available and was taken from GitHub (https://github.com/Alescontrela/Numpy-CNN). 

![image](https://github.com/Kaitan1995/Expediting-Convergence-via-Polling-Optimisation-for-Gradient-Descent-in-Neural-Networks/assets/93040738/45259adf-1b9a-423a-bdb3-df4228d86311)

![image](https://github.com/Kaitan1995/Expediting-Convergence-via-Polling-Optimisation-for-Gradient-Descent-in-Neural-Networks/assets/93040738/b742d4d7-4054-40e0-b802-ac4e90da7ea0)

## Experimentation Results
Direct cost function analysis was used to perform comparison studies of neural networks that employed the Polling Method against neural networks that did not, with the consideration being the ability of the models to combat undershooting and overshooting during gradient descent of the cost function towards the global minima. The consideration being loss against epoch. For ANN, all training was performed with a maximum epoch of 100. 
- ANN with Polling Method – Model 1.
- ANN with Polling Method/Adaptive Moment Estimation, ADAM – Model 2.
- Ordinary ANN – Model 3.
- Ordinary ANN with Adaptive Moment Estimation, ADAM – Model 4.
- Convolutional Neural Network with Polling Method/Adaptive Moment Estimation, ADAM – Model 5.
- Convolutional Neural Network with Adaptive Moment Estimation, ADAM – Model 6. 

For CNN, training with a maximum epoch of 5, randomised training examples of 500 are used from the training dataset, where the CNN with Polling Method is used to make a prediction. The predicted class label output is captured, and compared against the correct class label output, where the performance of both models for predicting the class labels are collated.

All experimentations were performed with the following hardware considerations: A NVIDIA GeForce RTX 2080 Ti was utilised that ran on-site at University of Newcastle upon Tyne located in Singapore within the Singapore Institute of Technology @ Nanyang Polytechnic. This is configured with CUDA Cores: 2944, Tensor Cores: 368, Base Clock: ~1515 MHz, Boost Clock: ~1710 MHz, Memory: 8GB GDDR6, Memory Speed: 14 Gbps, Memory Bus: 256-bit and Memory Bandwidth: 448 GB/s. 

- Processor: Intel(R) Xeon(R) CPU E5-1607 v4 @ 3.10GHz
- Installed RAM: 64.0 GB
- Display Adapter: NVIDIA GeForce RTX 2080 Ti
- System: Windows 10 Pro. 64-bit operating system, x64-based processor
- Version: 22H2
- OS build: 19045.3693
- Experience: Windows Feature Experience Pack 1000.19053.1000.0
- CUDA build: 11.2.r11.2/compiler.29373293_0
- CUDA DNN build: 8.1.1

## ANN with Polling Method – Model 1
The ANN consists of initial weights and bias initialisation, forward pass, backpropagation, updating of weights and biases, 3x learning rates (α_1,α_2,α_3) of values (0.1,0.5,0.9).

### ANN with Polling Method cost function
![image](https://github.com/Kaitan1995/Expediting-Convergence-via-Polling-Optimisation-for-Gradient-Descent-in-Neural-Networks/assets/93040738/54d2b27b-b52d-44f8-89ec-5a81b956d4e0)

### ANN with Polling Method gradient descent (with the best accuracy identified with ✓)
![image](https://github.com/Kaitan1995/Expediting-Convergence-via-Polling-Optimisation-for-Gradient-Descent-in-Neural-Networks/assets/93040738/3236ceea-4584-44a1-9b72-972554ed317a)

## ANN with Polling Method/Adaptive Moment Estimation, ADAM – Model 2
This will consist of initial weights and bias initialisation, forward pass, backpropagation, updating of weights and biases by ADAM optimiser, 3x learning rates (α_1,α_2,α_3) of values (0.01,0.25,0.5).

### ANN with Polling Method/ADAM cost function
![image](https://github.com/Kaitan1995/Expediting-Convergence-via-Polling-Optimisation-for-Gradient-Descent-in-Neural-Networks/assets/93040738/69fd39bc-f05f-4896-ad1d-562d7879553f)

### ANN with Polling Method/ADAM gradient descent (with the best accuracy identified with ✓) 
![image](https://github.com/Kaitan1995/Expediting-Convergence-via-Polling-Optimisation-for-Gradient-Descent-in-Neural-Networks/assets/93040738/d848de97-3d20-4771-9505-40a8bc070e89)

## Ordinary ANN – Model 3
This will consist of initial weights and bias initialisation, forward pass, backpropagation, updating of weights and biases, 1x learning rate (α) of values (0.5).

### ANN cost function
![image](https://github.com/Kaitan1995/Expediting-Convergence-via-Polling-Optimisation-for-Gradient-Descent-in-Neural-Networks/assets/93040738/19ae0e63-55c6-4f9c-93f6-21f2c3abb688)

## Ordinary ANN with Adaptive Moment Estimation, ADAM – Model 4
This will consist of initial weights and bias initialisation, forward pass, backpropagation, updating of weights and biases by ADAM optimiser, 1x learning rate (α) of values (0.01)).

### ANN with ADAM cost function
![image](https://github.com/Kaitan1995/Expediting-Convergence-via-Polling-Optimisation-for-Gradient-Descent-in-Neural-Networks/assets/93040738/f8099bd2-087b-4fb8-8694-8f0d6da73c0d)

## Convolutional Neural Network with Polling Method/Adaptive Moment Estimation, ADAM – Model 5
This will consist of initial weights and bias initialisation, convolutional initialisation, convolutional operations (maxpooling and flattening), forward pass, backpropagation, updating of weights and biases by ADAM optimiser, 3x learning rates (α_1,α_2,α_3) of values (0.01,0.05,0.1).

### CNN with Polling Method/ADAM cost function
![image](https://github.com/Kaitan1995/Expediting-Convergence-via-Polling-Optimisation-for-Gradient-Descent-in-Neural-Networks/assets/93040738/6e5f4772-54a6-4469-bb20-88a4e7214f84)

## Convolutional Neural Network with Adaptive Moment Estimation, ADAM – Model 6 
This will consist of initial weights and bias initialisation, convolutional initialisation, convolutional operations (maxpooling and flattening), forward pass, backpropagation, updating of weights and biases by ADAM optimiser, 1x learning rate (α) of values (0.01).

### CNN with ADAM cost function
![image](https://github.com/Kaitan1995/Expediting-Convergence-via-Polling-Optimisation-for-Gradient-Descent-in-Neural-Networks/assets/93040738/73136505-1527-49c5-a323-b6a702c14bb6)

# Results and Findings

For ANN Models, a direct comparison between Model 1 and Model 3 determines that there is comparable performance in terms of accuracy. With Model 2, the accuracy was raised to 89.3%, which while comparable to Model 4 alone, but outperforming Model 4 in terms of total raw losses in cost function analysis; Model 2 boasts a total raw losses score of ~5000 absolute difference error in Figure 4 against Model 4’s absolute difference error of ~10,000 in Figure 6, where the (collation over the epoch/iteration) absolute difference error is defined as the of the partial derivative of dA_2  from the loss function, dZ_2 which was derived from the one-hot encoding process as described in the backpropagation step. With the Polling Method (Model 2), the collective error is the lowest out of the comparable ANN models (Model 1, 3 and 4) using the same number of epochs/iterations. Furthermore, Model 2 appears to reach a point of convergence/saturation compared to the other ANN models.

An analysis of a CNN model with the Polling Method (Model 5) against a CNN model with ADAM optimiser (Model 6) strongly suggests a similarity in performance.


