# **Hand Gesture Recognition for Online Learning using MediaPipe**


---



The aim of this Project is to detect and recognise the most common Hand Gestures expressed in Online Learning using Python and [MediaPipe](https://google.github.io/mediapipe/solutions/hands). The model is trained to Recognize 4 Gestures expressing: 'Tiredness', 'Sickness', 'Critical Thinking' and 'Asking Questions' as demonstrated below:

![alt text](https://github.com/aminefarez/Hand-Gesture-Recognition_MediaPipe/blob/main/classes.JPG)

The repository includes:

*   Source code of Hand Gesture Recognition based on MediaPipe with pre-trained encodings for the 4 Gestures of Interest.

*   Training code to be used to train on your own dataset.

The code is documented and designed to be easy to extend. If you use it in your research, please consider citing this repository (bibtex below).

# **Prerequisites**
---





 


The libraries needed can be found in the [requirements.txt](https://github.com/aminefarez/Hand-Gesture-Recognition_MediaPipe/blob/main/requirements.txt) file, they can be installed using:



```
# pip install -r requirements.txt
```
Or if you're using [Google Colab](https://colab.research.google.com/):



```
# !pip install -r requirements.txt
```





# **Getting Started**


---






*   [hand_recognition.py](https://github.com/aminefarez/Hand-Gesture-Recognition_MediaPipe/blob/main/hand_recognition.py) Is the easiest way to start. It shows an example of using a pre-trained model to be used on a Video or Webcam Input.

*   [hand_training.py](https://github.com/aminefarez/Hand-Gesture-Recognition_MediaPipe/blob/main/hand_training.py) shows how to train the model on your own dataset.


