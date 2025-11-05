import cv2
import numpy as np
import math
import os
import time
import csv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0) # opens the cameraer
counter = 0
frame_count = 0
save_path = "eye_images"
model = load_model("gaze_cnn_model.h5")
class_labels = ["up_left", "up_right", "down_left", "down_right", "center"]


os.makedirs(save_path, exist_ok=True)
for label in class_labels:
    os.makedirs(os.path.join(save_path, label), exist_ok=True)

metadata = []
while True:
    ret, frame = cap.read() # reads frames from the camera

    # Calculate the dimensions of each quarter
    frame_height, frame_width = frame.shape[:2]
    quarter_height = frame_height // 2
    quarter_width = frame_width // 2

    # Split the frame into four equal parts
    top_left = frame[0:quarter_height, 0:quarter_width]
    top_right = frame[0:quarter_height, quarter_width:frame_width]
    bottom_left = frame[quarter_height:frame_height, 0:quarter_width]
    bottom_right = frame[quarter_height:frame_height, quarter_width:frame_width]

    if not ret: # if no frame is captured ( if cam closes or deactivates smhow) finish the loop 
        break
    
    # Assume you have cropped eye region (eye_frame)
    eye_frame = cv2.resize(frame, (64, 64))
    eye_input = np.expand_dims(eye_frame, axis=0) / 255.0

    pred = model.predict(eye_input)
    label = class_labels[np.argmax(pred)]

    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Gaze Prediction", frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Histogram Equalization
    gray_eq = cv2.equalizeHist(gray)  
    
    # eyes_cascade has the contents of an .xml file, ,
    # which is a code specially designed for eye and iris detection
    # named haarcascade_eye.xml. (yes, we fetched it from a GitHub page which will has its link in our appendices)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)
    
    # This way, we can detect the eye regions (yes, not iris) at the given frame (from camera, remember)
    for (x, y, w, h) in eyes:
        counter = counter + 1
        # after detecting the eye regions, we simply draw rectangles around these eye regions
        new_w = int(w * 0.5)
        new_h = int(h * 0.5)
        x += int((w - new_w) / 2)
        y += int((h - new_h) / 2)
        w = new_w
        h = new_h
        
        #Eye region crop
        eye_frame = frame[y:y + h, x:x + w]
        
        # then we extract the eye regions from the grayscale frame
        eye_gray = gray[y:y+h, x:x+w]
        
        # applies adaptive tresholding to separate irises from the rest of the eye
        # this creates an iris region as a result
        adaptive_th = cv2.adaptiveThreshold(eye_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)

        # applies morphology operations to remove noise and fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        iris = cv2.morphologyEx(adaptive_th, cv2.MORPH_OPEN, kernel)

        
        #finds contours in the iris region (iris region is created above)
        contours, _ = cv2.findContours(iris, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        for contour in contours:
            # Filters out contours based on aspect ratio and area
            x_contour, y_contour, w_contour, h_contour = cv2.boundingRect(contour)
            aspect_ratio = float(w_contour) / h_contour
            contour_area = cv2.contourArea(contour)
            
            # Adjusts these thresholds according to your specific requirements
            if aspect_ratio > 0.2 and aspect_ratio < 1.8 and contour_area > 100:
                valid_contours.append(contour)

        if len(contours) > 0:
            # finds the largest contour in the iris region
            iris_contour = max(contours, key=cv2.contourArea)
            
            #then we findd the center of the iris contour
            moments = cv2.moments(iris_contour)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                inner_radius = int(w/8)
                # draws iris  and pupil frames
                cv2.circle(frame, (x+cx, y+cy), inner_radius, (0, 0, 255), 2)
                # Estimate gaze direction based on previous and current coordinates
            
                 # Determine the position of the eye
                eye_position = "None\nNone"
                if x > quarter_width and y > quarter_height:
                    eye_position = "Down\nRight"
                elif x <= quarter_width and y > quarter_height:
                    eye_position = "Down\nLeft"
                elif x > quarter_width and y <= quarter_height:
                    eye_position = "Up\nRight"
                elif x < quarter_width and y < quarter_height:
                    eye_position = "Up\nLeft"
                position_lines = eye_position.split("\n")

                cv2.putText(frame, position_lines[0], (x - 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, position_lines[1], (x - 20, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Determine label for storage
                eye_label = "center"
                if x > quarter_width and y > quarter_height:
                    eye_label = "down_right"
                elif x <= quarter_width and y > quarter_height:
                    eye_label = "down_left"
                elif x > quarter_width and y <= quarter_height:
                    eye_label = "up_right"
                elif x < quarter_width and y < quarter_height:
                    eye_label = "up_left"

                # Save eye image with metadata
                timestamp = int(time.time())
                filename = f"{eye_label}_{frame_count}_{timestamp}.jpg"
                filepath = os.path.join(save_path, eye_label, filename)
                cv2.imwrite(filepath, eye_frame)
                metadata.append([filename, eye_label, timestamp])
                print(f"Saved: {filepath}")

        frame[y:y + h, x:x + w] = eye_frame
    if counter == 0:
        cv2.putText(frame, "No eyes detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        stringer = "Detected Eye #: " + str(counter)
        cv2.putText(frame, stringer, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    counter = 0
    # displays result
    cv2.imshow('Eye Detection and Iris Tracking', frame)

    # if the 'q' or 'Q' keys are pressed, breaks the loop
    # which closes the program
    # wait for user input to close the program
    key = cv2.waitKey(1)

    if key == 27:  # 27 is the ASCII code for the 'Esc' key
        break
        
    elif key == ord('q') or key == ord('Q'):
        break
    frame_count += 1
 
cap.release()  
cv2.destroyAllWindows()

if metadata:
    metadata_path = os.path.join(save_path, "metadata.csv")
    write_header = not os.path.exists(metadata_path)
    with open(metadata_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["filename", "label", "timestamp"])
        writer.writerows(metadata)
    print(f"Metadata saved to {metadata_path}")
