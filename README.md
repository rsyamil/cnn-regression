# cnn-regression

This is a simple guide to a vanilla convolutional neural network for regression, potentially useful for engineering applications and is intended for beginners. 

## Convolutional neural network (CNN) for regression

In [this Jupyter Notebook](https://github.com/rsyamil/cnn-regression/blob/master/cnn_regression.ipynb), we will first download the digit-MNIST dataset from Keras. This dataset comes with a label for each digit and has been widely used for classification problem. In many engineering problems however, we may need to do more than classification.

![Dataset](/readme/dataset.png)

To demonstrate how we can use a convolutional neural network to perform a regression task, we first assume an operator **G** that we will use as a forward model on each of the MNIST images, to simulate a vector of observations. In some engineering applications, the MNIST images may represent physical model of subsurface systems **X** and the operator **G** may represent a multiphase flow simulator. The collected or simulated data, **Y** from each of the MNIST images represents the observations or response. It can be written in a general form as **Y=G(X)** and for the purpose of this demonstration, let us consider the linear version where **Y=GX**.

![ForwardModel](/readme/forwardmodel.png)

In the plots below, the responses from each class of MNIST digits are shown. Simply put, the operator **G** simulates arrival times of rays that are transmitted from the left and  top sides of an image and received on the right and lower sides respectively. Each vector of simulated arrival times contains important information about the image that is used to generate the observation. For example, the simulated responses for digit *0* are almost symmetrical about half of the x-axis as each digit *0* will generate similar response when the rays are transmitted left-to-right or top-to-bottom. The width of the digit in the image will also affect the simulated responses. Note that this operator **G** can be replaced by any physical simulator. 

![SimulatedY](/readme/simulatedys.png)

The complexity of the images and the linear forward model are captured using a convolutional neural network. The CNN can also be utilized to represent the relationship between input and output data with unknown physical equations. 2D convolutions are used on the images to extract salient spatial features and multiple dense layers are combined with the convolutional filters. 

![Architecture](/readme/architecture.png)

Once trained, the resulting CNN can be used to provide a response prediction for any given MNIST image. Such predictive model is also usually referred to as a proxy model, a meta-model or a surrogate model and can be highly useful when a single run of the operator **G** can take hours or even days! Here the convolutional filters for the trained proxy model are visualized.   

![Filters](/readme/trained_filters.png)

The filter activations (or intermediate representations) from the trained CNN, for a sample image from the test dataset are shown here. Similar to the [classification problem](https://github.com/rsyamil/cnn-classifier), the convolutional filters extract salient spatial features from the (somewhat redundant) images. Unlike the classification model where the combination of these features is used to distinguish between the labels, for a regression problem, the combination of these features is used to predict the response. 

![Activations](/readme/activations.png)

Some samples of test images with their associated response predictions are shown below. Overall the predictions are satisfactory and agree with the *true* responses. From the plots below, we can notice that each response has key signatures resulting from the spatial features present in each digit image.    

![PredMatch](/readme/predictionmatch.png)

In the architecture of the CNN used in this demonstration, the first *Dense* layer has an output dimension of 16 to give satisfactory predictive capability. In the [classification problem](https://github.com/rsyamil/cnn-classifier) considered previously, the first *Dense* layer has an output dimension of only two. This difference provides an insight on the complexity level of a classification problem versus a regression problem. Below the activations of the first *Dense* layer, for each of the 16 output variables are plotted and color-coded by digit labels.   

![ActivationsDenseNN](/readme/dense_activations_nn.png)

Next, let's run a quick experiment to see if a regression model based on CNN can be utilized for transfer learning, since most transfer learning applications are for classification problems. We will pre-train a regression CNN with images of digit *8* and *9* with the corresponding simulated responses.

![DatasetTransfer](/readme/dataset_transfer.png)

Then using the pre-trained model, the weights for the convolutional filters are locked and the weights for the *Dense* layers are allowed to be optimized. Images of digit *2* and the corresponding simulated responses are used as the test/transfer dataset. 

![ArchitectureTransfer](/readme/architecture_transfer.png)

The plots below show some examples of test cases. In general, the predictions from a "transfer" model (i.e. pre-trained CNN that is re-trained with data from digit *2*) show better match with the true case. This should not come as a surprise since the re-trained CNN has had the opportunity to learn from the training data that includes **X** and **Y** from digit *2*.

![PredMatchTransfer](/readme/predictionmatch_transfer.png)

In practical applications, the knowledge to be transferred may represent complex physical equations with varying initial/boundary conditions. We also may not have sufficient test or validation data. Transferring relevant knowledge from appropriate dataset may help a predictive model generalize better for unseen data. 
