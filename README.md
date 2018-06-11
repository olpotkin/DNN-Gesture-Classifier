# Deep Neural Network Gesture Classifier

## Static gestures classification using Convolutional Neural Network on the example of the Russian Sign Language

[image1]: ./imgs/01_dataset.png "dataset"
[image2]: ./imgs/02_segment.png "segment"
[image3]: ./imgs/03_cnn.png "cnn"
[image4]: ./imgs/04_train.png "train"
[image5]: ./imgs/05_exp.png "exp"
[image6]: ./imgs/06_res.png "res"
[image7]: ./imgs/07_cnn.png "cnn"


## Domain Background
Interfaces of human-computer interaction are diverse in their implementation and scope e.g. systems with console input-output, controllers with gesture control, brain-computer interfaces and others. Systems using data input based on the recognition of custom gestures have gained wide popularity since 2010 after the release of the contactless game controller Kinect from Microsoft. Gesture based controllers increase their market share and become a part of everyday life of different categories of users. So, for example, the Volkswagen car manufacturer introduced the multimedia system Golf R Touch Gesture Control to control the multimedia system of the car by gesture commands.

To translate gesture commands into a control signal, a gesture classification mechanism is needed, which can be obtained from various devices: special gloves defining joint coordinates, as well as 2D and 3D video cameras. The approach using gloves has a significant drawback – a user needs to wear a special device connected to the computer. In turn, the approach based on the concept of computer vision using video cameras is considered more natural and less expensive.


## Problem Statement
Project demonstrates the core of Russian Sign Language static gesture classification system, which is based on the approach of computer vision with using convolutional neural network. The work is actual and represents a starting point for researchers in the field of gesture recognition.


## Datasets and Inputs
To solve described problem with CNN approach I need a dataset. I didn’t found any open dataset in this field and decided to create my own.

The dataset for learning, validating and testing a neural network consists of **around 1000 images** (1042 on the moment of writing of this paper) with a resolution of **128x128 pixels**. The image is divided into **10 classes**, each of which corresponds to a strictly defined gesture.

Dataset is a part of repository: hand-dataset.zip

Dataset stats:

![alt text][image1]

The image is divided into 10 classes, each of which corresponds to a strictly defined gesture (picture 1): 
- Line 1 – Class Id.
- Line 2 – Value / Letter.
- Line 3 – Quantity in dataset.
- Line 4 – image with gesture example.


## Solution Statement
- Step 1 – image preprocessing. In this step, I’m going to check preprocessing techniques to apply it for an image before providing it for CNN.
- Step 2 – normalization. Each pixel value must be in the range 0 to 1 (for Keras input).
- Step 3 – training/validation/test split of the dataset. Here I’ll need to choose a proportion.
- Step 4 – define CNN architecture. In this step, I need to check different CNN architecture and to see which one performs better.


## Data preprocessing
Data preprocessing includes 2 steps:

1. Convert RGB to YCrCb color space
2. Threshold color components to perform hand segmentation

```
skin_ycrcb_mint = np.array((0, 133, 77))
skin_ycrcb_maxt = np.array((255, 173, 127))
```

Result after preprocessing:

![alt text][image2]

After preprocessing we need to split dataset to the 3 sets:

* Training set: 666
* Cross-validation set: 167
* Test set: 209


## Benchmark Model
In this project, I decide to use the state-of-the-art benchmark model (architecture) – LeNet-5 as a BASE for my CNN architecture. It shows impressive result in image classification tasks and has very good performance.


## Neural Network Architecture
The LeNet-5 architecture taken as a base. The final model architecture consisted of a convolutional neural network with the following layers and layer sizes:

![alt text][image3]

![alt text][image7]

Important design choice - to apply Dropout - a simple way to prevent neural networks from overfitting).


## Evaluation Metrics
- Training. Evaluating parameters – accuracy, loss.
- Validation. Performs every epoch of CNN training process. Evaluating parameters – accuracy, loss.
- Test. Performs after the training of CNN on the data that not presented during the training process. Evaluating parameter – accuracy.


## Training results
The validation set helped determine if the model was over or under fitting:

![alt text][image4]

And, finally, test set results:

* Test loss: 0.238594918445
* Test accuracy: 0.913875598657


## Experiment
Evaluate the data that never been in the dataset:

![alt text][image5]

The probability distribution for 10 examples:

![alt text][image6]


## Evaluation
* Class 0 (А): True, Probability = 1
* Class 1 (Б): True, Probability = 0.998
* Class 2 (В): True, Probability = 0.998
* Class 3 (Г): **False**, Probability = 0 (Detected Class 7 with Probability = 1)
* Class 4 (Е): **False**, Probability = 0.021 (Detected Class 7 with Probability = 0.783)
* Class 5 (И): True, Probability = 0.94
* Class 6 (О): True, Probability = 1
* Class 7 (П): True, Probability = 1
* Class 8 (С): True, Probability = 1
* Class 9 (Я): True, Probability = 0.997

---

* 91.3% of images **from the test set** were correct classified.
* 80% of images **from the out of the dataset** were correct classified.


## Possible points of improvement
- Use much more data (to think about croudsourcing data collection)
- Use data augmentation techniques
- Use advanced preprocessing techniques (more experiments with color spaces, hog features etc.)
