# Deep Neural Network Gesture Classifier

## Static gestures classification using Convolutional Neural Network on the example of the Russian Sign Language (RSL) dactyl

[image1]: ./imgs/01_dataset.png "dataset"
[image2]: ./imgs/02_segment.png "segment"
[image3]: ./imgs/03_cnn.png "cnn"
[image4]: ./imgs/04_train.png "train"
[image5]: ./imgs/05_exp.png "exp"
[image6]: ./imgs/06_res.png "res"
[image7]: ./imgs/07_cnn.png "cnn"

The project demonstrates the system of classification of static gestures of RSL, which is based on the approach of computer vision using convolutional neural network. The work is actual and represents a starting point for researchers in the field of gesture recognition.

---

## Dataset

The dataset for learning, validating and testing a neural network consists of **around 1000 images** (1042 on the moment of writing of this paper) with a resolution of **128x128 pixels**. The image is divided into **10 classes**, each of which corresponds to a strictly defined gesture.

Dataset stats:

![alt text][image1]

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

After preprocessing we need to split dataset to:

* Training set: 666
* Cross-validation set: 167
* Test set: 209

## Neural Network Architecture

The final model architecture consisted of a convolutional neural network with the following layers and layer sizes:

![alt text][image3]

![alt text][image7]

Important design choice - to apply Dropout - a simple way to prevent neural networks from overfitting).

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

### Evaluation

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
* 80% of images **out of the dataset** were correct classified.

