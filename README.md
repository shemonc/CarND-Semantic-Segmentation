#TBD


# Semantic Segmentation
### Introduction
Goal of this project, is to label the pixels of a road in images using a Fully Convolutional Network (FCN) for Semantic  Segmentatin basd on a VGG net (encoder and decoder) developed [here ](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf), train and test this model on KITTI data set

### Architecture

A fully convolutional version of VGG16 (ENCODER), which already contains the 1x1 convolutions to replace the fully connected layers in order to preserve the spatial information.

### Decoder implementation FCN8
1. input the last layer of vgg16 encoder which is layer 7, shape [5 5 18 4096]
2. 1x1 convolution of vgg encoder layer 7
    The pretrained VGG-16 model is already fully convolutionalized, i.e. it
    already contains the 1x1 convolutions that replace the fully connected
    layers. THOSE 1x1 convolutions are the ones that are used to preserve
    spatial information that would be lost if we kept the fully connected
    layers. The purpose of the 1x1 convolutions that we are adding on top
    of the VGG is merely to reduce the number of filters from 4096 to the
    number of classes which 2(road or not road) for our model.
    output shape [5 5 18 2]
3. Upsample the output from step 2 above with a strides of 2 and kernel size of 4.
   output shape [5 10 36 2] for this layer
4. Apply 1x1 convolution of vgg pool4 layer to match the shape with above upsample layer
   which is [5 10 36 2]
5. Add convoluted vgg pool4 with upsample layer at step 3. above, this is calld skip connection.
   Skip connections allow the network to use information from multiple resolutions, as a result
   the network is able to make more precise segmentation decision.
6. Second Upsample, input [5 10 36 2] output [5 20 72 2]
7. 1x1 convolution of vgg pool3 layer to match the shape with above upsample layer,
   after convolution output shape [5 20 72 2]
8. Another skip connection between 6 and 6 above, output shape [5 20 72 2]
9. Final upsample by 8x8 stride
   input shape [5 20 72 2], output shape [5 160 576 2] <-- original image size

### Optimizer

I use cross-entropy loss and l2 regularization loss with an Adam optimizer

### Training

#### Hyperparameters
1. learning rate: 0.0001
2. keep prob: 0.5 during training.
3. Epochs: 30
4. Batch size: 5
5. l2 regularization at a scale of 1e-3 and a factor of 0.01 of total regularization loss

### Results

Epoch 1  
loss  
mean 0.629  
std 0.289  

Epoch 2  
Loss  
mean 0.199  
std 0.042  

Epoch 10  
loss  
mean 0.055  
std 0.013  

Epoch 20  
loss  
mean 0.038  
std 0.007  

Epoch 30  
loss  
mean 0.025  
std 0.005  

### Sample Classified Images

![sample1]:(./images/um_000010.png)
![sample2]:(./images/um_000019.png)



### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
[Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
