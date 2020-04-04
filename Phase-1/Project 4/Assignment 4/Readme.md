# Architectural Basics

## 1. [Architecture of the neural network.](https://github.com/LokeshJatangi/EVA-vision/blob/master/Project%204/Assignment%204/code%20-1.ipynb)

The Sequential neural network architecture for the Assignment 4 is as follows:

 
>_Convolution Block 1 Begins_   
**28x28x1  | (3x3)x10 | 26x26x10   , Receptive Field = 2x2**  
**26x26x10 | (3x3)x10 | 24x24x10   , Receptive Field = 4x4**  
**24x24x10 | (3x3)x30 | 22x22x30   , Receptive Field = 6x6**   
>_Convolution Block 1 Ends_
*** 
>_Transitional Block  Begins_  
**Max Pooling  (2x2)               , Receptive Field = 12x12**  
**11x11x30 | (1x1)x10 | 11x11x10   , Receptive Field = 12x12**  
>_Transitional Block  Ends_
***  
>_Convolution Block 2 Begins_  
**11x11x10 | (3x3)x10 | 9x9x10     , Receptive Field = 14x14**  
**9x9x10   | (3x3)x18 | 7x7x18     , Receptive Field = 16x16**  
**7x7x18   | (3x3)x30 | 5x5x30     , Receptive Field = 18x18**  
>_Convolution Block 2 Ends_ 
***
>_Transitional Layer Begins_  
**5x5x30   | (1x1)x10  | 5x5x30    , Receptive Field = 18x18**  
>_Transitional Layer Ends_  

>**5x5x30   | (5x5)x10  | 1x1x10    , Receptive Field = 22x22**  
*** 

### 1.Image Normalization ###  
>Considering Image Normalization as Pixel Normalization, Each pixel is divided by factor 255 which changes the range of pixel values in (0,1). 
### 2.Kernels  (Feature extractors) ###
 >The Kernels applied in the network is 3x3, which is the most popular filter size.A small filter size of 3x3 is sufficient for a MNIST data set as the amount of information present in the images is low and as well the hardwares are optimized for 3x3 kernels.
### 3.Number of Layers ###
 >Intially the number of layers were added till Receptive field reached around 24x24 whilst compacting Covolutional layers into  Convolutional blocks and Transitional Blocks.  After few trials of different architectures the above architecture was finalised and here are the details:     
>> Convolutional Blocks : 2   
>> Transitional Block : 1  
>> Transitional layer : 1  
>> Layer with a Large Kernel: 1  

>Each Convolutional block has 3 Convolutional layers, Hence the Total number of layers would be : 2x3 + 1 + 1 + 1 =9 layers.

### 4.Max Pooling and its Position 
> Max Pooling identifies the most important features of the image and reduces the dimension of image by half when its size is (2x2) and would increase the Receptive field by a factor of 2.     
The position of the Max pooling is decided on the basis of when the kernels start to form textures and gradients , As we see in the MNIST data set the image sizes are less so the Textures and gradients would be created after two or three Convolution layers.  
After Max pooling only 25% of the information is retained  
 
### 5.Distance of Max pooling from prediction
> Max pooling is not added when it is three or four layers away from prediction , Hence after the second block of convolution the Max pooling was not added and only the Transitional layer was added.

### 6. Transitional Layer 
> The concept of transitional layer or 1x1 convolution is that it combines the extracted features in the previous layer and creates new feature.The number of kernels used in this layer must be less than the previous Convolution layer only then new feature maps are created.Thus it reduces the number of channels inturn which reduces the number of parameters.

### 7. Position of Transitional layer 
> The Position of Transitional layer could after or before the Max Pooling layer and it should be used when large number of kernels are applied in previous Convvolution layer.Here I have used after Max pooling.

### 8. Receptive Field 
>The receptive field is part of the image that is visible to the filters or it is part of image the network has seen . In this network as I have used 3x3 kernels the receptive field would be 2x2 .
After Max pooling the Receptive field increases by factor of 2. 

### 9.How do we decide the Number of kernel ?
> The number of kernels was decided keeping mind that total number of parameters should be less than 15k.  
As the number of class variations is 10 ,Thus the first layer was started with 10 number of convolutions and it was increased to 30 as the convolution block ends so that when transitional layer is applied in next block network would have more number of features to extract and combine it back to 10 channels for next convolution block to begin.  
In the second convolution block the second layer has 18 kernels (instead of 10 kernels from Convolution block 1) because to provide more expressivity. (Intially it was 20 kernels but the parameters overshot 15k , so it was reduced by 2 to reach 18 kernels).
After the convolution block 2 ended an additional Transitional layer is added.    
{The number of kernels was concluded based on the few trial runs.}

### 10. When do we stop convolutions and go ahead with larger kernel ?
>The larger kernel could be applied after few Convolution and Transitional blocks, Here it was added and examined for three input sizes  5x5 , 3x3 .
The validation accuracy was more for 5x5 than 3x3 input size in this case and the number of parameters exceeded 15k when Convolution layers were added to reach in put size of 3x3.

#### 11. Softmax
>Softmax is a normalized exponential function which converts the input vector into representation of categorical distribution which sums up to 100%.

### 12.Validation Checks
> Validation checks are added for each epoch.
It would help us determine when the validation accuracy was highest out of all epochs.

### 13.Optimizer
>Optimizer applied in the network is **ADAM**.

### 14.How is best network decided in the early stages ?
>In all the different trial runs of the Neural network architectures by changing the number of layers in Convolution block , Number of kernels in each layer.    
The best network is found by noting down training accuracies for first few epochs in each trial.
The potential of the network should also be kept in mind ,it has calculated as a difference between training and validation accuracy.

**Conclusion after First code**
>As the network is overfitting,In the next iteration the changes that would be added are Batch Normalization  

## 2.[Batch normalization and Batch size](https://github.com/LokeshJatangi/EVA-vision/blob/master/Project%204/Assignment%204/Code-2.ipynb)

### 1. Batch Normalization 
> The Batch Normalization standardizes distribution of features after a Convolution layer is applied as some kernels would have higher activations than other kernels.  
In this network Batch Normalization is applied after each Convolutional layer.

### 2. Distance of Batch Normalization from prediction
> Batch normalization is not just before layer which has a larger kernel size or just before the last layer.

### 3. Batch size and effects of batch size
>Batch size determines the number of input images randomly passed to Network on which the forward pass occurs in each epoch for pre-determined number of iterations.  
The effects of batch size is visible on duration of each epoch as it gets reduced for higher batch sizes.    
The best batch size is found out by experimenting with different values and noting down accuracies in each experiment.

**Conclusion after SECOND code**
>The training accuracy has increased , but the overfitting problem is not solved , Hence dropouts are added.  
The best batch size for the second network was found to be 512 from experimenting with it.

## 3.[DropOut](https://github.com/LokeshJatangi/EVA-vision/blob/master/Project%204/Assignment%204/Code-3.ipynb)

### 1.DropOut 
> Dropout means randomly ignoring the effect of neurons during training phase. At each training phase few(depends on the droprate parameter) neurons are dropped.It helps the network to learn more robust features and regularizes the network.IIt reduces interdependent learning amongst the neurons.     
The droprate used is 10% after each convolution layer except the last layer before prediction.

### 2.When do we introduce DropOut, or when do we know we have some overfitting ?
>To prevent overfitting of model the dropOuts are used.  When the difference between validation accuracy and Training accuracy is not low, then it is called a overfitting model.

**Conclusion after THIRD code**
> Here the problem of overfiting is surpassed by adding the dropouts , the next agenda is to reach a validation accuracy 99.4%.So, I would trying out the Learning rate scheduler to obtain the required validation accuracy within 20 epochs.  

## 4.[Learning rate Scheduler]()

### 1.Learning rate
>During BackPropagation,when the weights of the neural network are altered based on the findings of the Minimum value of Loss function through the stochastic gradient descent , The step size in the search of the minimum value Error function.  
It controls the rate at which model learns.

### 2. Learning Rate scheduler 
>Learning rate scheduler is a function that takes learning rate and epoch index as parameters and provides the new learning parameter.  
The learning rate changes after each epoch in order to reduces the number of computations required to find minimum loss.

**Conclusion after Fourth and final code**
> The validation accuracy was found to be **99.42% in 17th epoch** for a total 14,820 parameters.

















