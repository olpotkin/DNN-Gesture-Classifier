# Deep Neural Network Gesture Classifier

## Static gestures classification using Convolutional Neural Networks on the example of the Russian Sign Language (RSL) dactyl

[image1]: ./imgs/01_dataset.png "dataset"
[image2]: ./imgs/02_segment.png "segment"
[image3]: ./imgs/03_cnn.png "cnn"
[image4]: ./imgs/04_train.png "train"
[image5]: ./imgs/05_exp.png "exp"
[image6]: ./imgs/06_res.png "res"

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

![alt text][image1]









