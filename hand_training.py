# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 11:51:53 2021

@author: Mohammed Amine Farez
"""

# Import the necessary libraries:
import cv2
import mediapipe as mp
import numpy as np

# Initialise MediaPipe solution for Hand Tracking:
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Initialise Parameters:
count = 0
processed_frame = 0
c = []

# For webcam input:
    #r'THE PATH TO THE FILE OR DATASET'

# Initialise the list to calculate the mean of encodings ( Sum / Number of frames processed ):
x_sum = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
y_sum = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# Read from the Path of the dataset of video file:
cap = cv2.VideoCapture(r'.\Data\*.mp4')

# Obtain the number of frames in Video:
len_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Launch the MediaPipe Hand Tracking with 0.5 Confidence:
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    # Iterate over each Frame:
    while cap.isOpened():
        success, image = cap.read()
        count += 1
        
        # Neglect the first 30 Frames for better accuracy:
        if count < 30:
            continue
        
        # Break if the Video comes to an end:
        if count == len_frames:
            break
        
        # Neglect empty Frames:
        if not success:
            print("Ignoring empty camera frame.")
            continue
    
        # Mark the image as not writeable to improve performance:
        image.flags.writeable = False
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
    
        # Draw the hand annotations on the image:
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Initialise the lists for 2D pixel values:
        x_encoding = []
        y_encoding = []
        lml = []
        xl = []
        yl = []
        
        # When the Hand is detected:
        if results.multi_hand_landmarks:
            # Draw the Landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            # Transform the 3D points to 2D pixel values:
            for id, lm in enumerate(results.multi_hand_landmarks[0].landmark):
                #  Obtain the Height of Width of the Frame:
                h, w, _ = image.shape
                # Append the x,y coordinates for each point of the landmark:
                xc, yc = int(lm.x * w), int(lm.y * h)
                lml.append([id, xc, yc])
                xl.append(xc)
                yl.append(yc)
    
            # Calculate the distances between x values, and between y values:
            for i in range(0,len(xl)-1):
                x_encoding.append(abs(xl[i]-xl[i+1]))
                y_encoding.append(abs(yl[i]-yl[i+1]))
    
            processed_frame += 1
            
            # Add the encoding of this frame to the Sum of encodings:
            x_sum = np.add(x_sum, x_encoding)
            y_sum = np.add(y_sum, y_encoding)
            
        # Flip the image horizontally for a selfie-view display:
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
     
        # Break when Q is pressed:
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Calculate the mean of the x and y encodings:
x_sum[:] = [x / processed_frame for x in x_sum]
y_sum[:] = [x / processed_frame  for x in y_sum]
print(" Average x = ", x_sum)
print(" Average y = ", y_sum)

# Destroy windows and release the camera when the process has finished:
cv2.destroyAllWindows()
cap.release()