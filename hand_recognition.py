# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 23:00:03 2022

@author: Mohammed Amine Farez
"""

# Import the necessary libraries:
import cv2
import mediapipe as mp
import math

# Initialise MediaPipe solution for Hand Tracking:
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# Import the Encodings of the Gestures of Interest:
Question =  [71.84, 69.43, 51.87, 38.95, 85.09, 67.78, 40.8, 37.11, 147.56, 77.88, 47.3, 42.2, 182.02, 72.99, 44.72, 39.05, 183.7, 57.56, 34.93, 32.28]
Tired =  [57.31, 55.15, 39.82, 30.89, 74.24, 62.97, 44.55, 35.51, 146.33, 73.68, 47.52, 38.47, 165.56, 71.7, 45.4, 36.35, 162.59, 54.74, 34.53, 28.28]
Critical =  [76.49, 68.45, 47.71, 32.65, 129.25, 66.1, 38.83, 33.73, 136.34, 77.67, 53.0, 34.79, 112.93, 79.16, 51.0, 33.11, 103.58, 67.12, 36.88, 24.6]
Sick = [60.22, 57.25, 44.72, 35.44, 66.71, 76.12, 45.88, 32.65, 141.6, 88.28, 47.76, 33.12, 158.07, 82.73, 46.17, 32.7, 156.7, 66.61, 37.66, 29.61]

encodings = [Question, Tired, Critical, Sick]
encodings_name = ["Question", "Tired", "Critical", "Sick"]


# For video input:
#cap = cv2.VideoCapture(r'THE PATH TO THE FILE')

# For webcam input:
cap = cv2.VideoCapture(0)

# Obtain the number of frames in Video:
len_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Count the frames:
count = 0

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
        
        # Initialise the list for the number of matches obtained for each Hand Gesture class: 
        matches = []
        
        
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
            
            # Calculate the Euclidean distance between each point of the Landmark:
            dist = [round(math.sqrt(i**2 +j**2),2) for i, j in zip(x_encoding, y_encoding)]
            
            # Compare the distances obtained with the pretrained encodings:
            for encoding in encodings:
                mat = 0
                
                # Normalize the distances by the first distance between point 0 and point 1:
                result = [round((i*encoding[0])/(j*dist[0]),2) for i, j in zip(dist, encoding)]
                
                # For each comparison, if the result is between 0.71 and 1.21, we have a match:
                for i in range(0,len(result)):
                    test = result[i]
                    if test < 1.21 and test > 0.71:
                        mat += 1
                        print(mat)
                
                # Append the number of matched distances with each Hand Gesture class:
                matches.append(mat)
        
            # Obtain the Class with the highest number of matches:
            gesture_class = max(matches)
            
            # Obtain the name of the Hand Gesture:
            hand_gesture = encodings_name[matches.index(gesture_class )] 
            
            # Display the result if at least Half the distances match (10/20):
            if gesture_class > 10:
                cv2.putText(image,hand_gesture,(100, 150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
        # Flip the image horizontally for a selfie-view display:
        #cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        cv2.imshow('MediaPipe Hands', image)
     
        # Break when Q is pressed:
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Destroy windows and release the camera when the process has finished:
cv2.destroyAllWindows()
cap.release()