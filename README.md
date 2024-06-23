# LAB: Learning States Feedback System

**Date**: 2022.06.22

**Name**: Moon Hyeon Ho

**Demo Video**: [Link](https://youtu.be/-X37Pc_AxDw)

**Github : **[Github](https://github.com/12-dimension-cat/DLIP_Project_Learning-States-Feedback-System_2024)

## I. Introduction

In this lab, we will use YOLOv8 and YOLOv8-Pose to record the study status while sitting, and create a system that informs how much time was spent studying after the session ends. To monitor the study status, we will classify into three states: studying, using a phone, and sleeping. The total sitting time will be set as the total time, and after the session ends, the system will display the proportion of each state relative to the total sitting time.



![image](https://github.com/12-dimension-cat/neko/assets/144550430/85f9414c-1dd7-4011-9d72-c470f19a6880)

<center>  Figure 1.  Example Image Output

## II. Requirement

### Hardware

- MechaSolurion Full HD 1080P Webcam

### Software

- Python 3.9.18
- numpy 1.26.4
- OpenCV 4.10.0
- PyTorch version: 2.1.2
- CUDA version: 11.8
- Yolov8
- Yolov8-pose



## III. Flow Chart

![image](https://github.com/12-dimension-cat/neko/assets/144550430/75dc4169-b122-40da-a236-1f5a43493dcf)

<center>  Figure 2.  Flow Chart



## IV. Procedure

### 1. Setup

First, installation is carried out using Anaconda Prompt to build the environment. It is important to install something suitable for each version using anaconda to build it to enable image processing.

### 2. Installation

#### 2-1. Install Anaconda

**Anaconda** : Python and libraries package installer.

Click here [Download the installer on window ](https://www.anaconda.com/products/distribution#Downloads)to Windows 64-Bit Graphical Installer

<img src="https://ykkim.gitbook.io/~gitbook/image?url=https%3A%2F%2Fgithub.com%2Fckdals915%2FDLIP%2Fblob%2Fmain%2Fsrc%2FDLIP_Project_ExercisePostureAssistanceSystem_2022%2Fpicture%2FAnaconda.jpg%3Fraw%3Dtrue&width=300&dpr=4&quality=100&sign=556ef013a22187b17115172a2708ecdc65d52e6d26c6cbed1f6bb155719b6025" alt="img" style="zoom: 33%;" />

<center>  Figure 3.  Type of Anaconda OS Installation

Follow the following steps

- Double click the installer to launch.
- Select an install for "Just Me"(recommended)
- Select a destination folder to install Anaconda and click the Next button.
- Do NOT add Anaconda to my PATH environment variable
- Check to register Anaconda as your default Python.

<img src="https://ykkim.gitbook.io/~gitbook/image?url=https%3A%2F%2Fgithub.com%2Fckdals915%2FDLIP%2Fblob%2Fmain%2Fsrc%2FDLIP_Project_ExercisePostureAssistanceSystem_2022%2Fpicture%2FAnaconda_1.jpg%3Fraw%3Dtrue&width=300&dpr=4&quality=100&sign=5cf98dee78b52f9281d88e64d537c98de332acee70338eb84f062619f5ade78a" alt="img" style="zoom:33%;" />

<center>  Figure 4. Anaconda installation screen

#### 2-2. Install Python

**Python 3.9**

Python is already installed by installing Anaconda. But, we will make a virtual environment for a specific Python version.

- Open Anaconda Prompt(admin mode)

![image](https://github.com/12-dimension-cat/neko/assets/144550430/172a5045-adaa-422b-967c-99f66eeb2c1d)

<center>  Figure 5. Anaconda Prompt



- First, update **conda** and **pip** (If the message "Press Y or N" appears, press Y and Enter key)

Copy

```
# Conda Update
conda update -n base -c defaults conda

# pip Update
python -m pip install --upgrade pip
```

<img src="C:\Users\moonh\AppData\Roaming\Typora\typora-user-images\image-20240622195111372.png" alt="image-20240622195111372" style="zoom:150%;" />

<center>  Figure 6. conda install

- Then, Create virtual environment for Python 3.9, Name the $ENV as `py39`. If you are in base, enter `conda activate py39`

Copy

```
# Install Python 3.9
conda create -n study_env python=3.9.18
```

![img](https://ykkim.gitbook.io/~gitbook/image?url=https%3A%2F%2Fgithub.com%2Fckdals915%2FDLIP%2Fblob%2Fmain%2Fsrc%2FDLIP_Project_ExercisePostureAssistanceSystem_2022%2Fpicture%2Fpython39_install.jpg%3Fraw%3Dtrue&width=300&dpr=4&quality=100&sign=d6f8b1c81ce3d3c5c3cc61d6e3fec1d9bffc1cdb0616f7997f7d44e10b5b22a4)

<center>  Figure 7. python installation screen

- After installation, activate the newly created environment

Copy

```
# Activate py39
conda activate study_env
```

![img](https://ykkim.gitbook.io/~gitbook/image?url=https%3A%2F%2Fgithub.com%2Fckdals915%2FDLIP%2Fblob%2Fmain%2Fsrc%2FDLIP_Project_ExercisePostureAssistanceSystem_2022%2Fpicture%2Fpy39.jpg%3Fraw%3Dtrue&width=300&dpr=4&quality=100&sign=9e587a7f60257ab0b2c039d64c1c84517476cb6ddb4bf5d5050de6043a00f6a9)

<center>  Figure 8. activate study_env



#### 2-3. Install Libs

**Install Numpy, OpenCV, Jupyter (opencv-python MUST use 4.10.0)**

Copy

```
conda activate study_env
conda install -c conda-forge opencv
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

#### 2-4. Install Visual Studio Code

Follow: [How to Install VS Code](https://ykkim.gitbook.io/dlip/installation-guide/ide/vscode#installation)

Also, read about

- [How to program Python in VS Code](https://ykkim.gitbook.io/dlip/installation-guide/ide/vscode/python-vscode)



#### 2-5. Install PyTorch and CUDA

**Install PyTorch 2.1.2 CUDA version: 11.8(If it's compatible with your graphics card)**

Copy

```
conda activate study_env
conda install pytorch torchvision torchaudio cudatoolkit=2.1.2 -c pytorch -c nvidia -y
```



#### 2-6. Install yolov8 and yolov8-pose

**If the environment(study_env) is running**

Copy

```
pip install ultralytics
pip install yolov8
pip install yolov8-pose
```



### 3. library you need

```
import cv2
from ultralytics import YOLO
import numpy as np
```



### 4. Global Variable

#### 4-1. Main Variable

**Definition Body Parts:** The output of the deep learning model we use is location information for 17 joints. The position of each joint has an order. The order is as follows.

[ nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle ]

Since we use the positions of the shoulders, elbows, and wrists on both sides in this application, we defined them as follows.

**Definition of body edges:** Each joint is tied together to draw a skeleton model.



**Flag: Conditions that set whether you study, using phone, or sleep.**

- flag_1: When the right wrist is lower than the elbow, and the elbow is lower than the shoulder. 

- flag_2: When the left wrist is lower than the elbow, and the elbow is lower than the shoulder. 

- flag_3: When the right elbow is lower than the wrist, and the wrist is lower than the shoulder. 

- flag_4: When the left elbow is lower than the wrist, and the wrist is lower than the shoulder. 

- flag_5: When the right wrist is more than 40 lower than the shoulder. 

- flag_6: When the left wrist is more than 40 lower than the shoulder. 

- flag_7: When the right elbow is higher than the shoulder, and the shoulder is higher than the wrist. 

- flag_8: When the left elbow is higher than the shoulder, and the shoulder is higher than the wrist. 

- flag_9: When the left wrist is to the right of the right wrist. 

- flag_10: When the distance between the right wrist and the left wrist is 250 or more. 

- flag_11: When the distance between the right wrist and the left wrist is 250 or less. 

- flag_12: When the left wrist is to the left of the right wrist. 

- flag_13: When the distance between the phone's position and the right wrist is less than 200. 

- flag_14: When the distance between the phone's position and the left wrist is less than 200. 

- flag_15: When the height difference between the keyboard position and the left wrist is less than 100.

  **(The numbers represent the difference in video frame values.) **

  

**paused** : Pause flag for bathroom breaks or urgent matters, and a flag to display the final results.

**For counting:** Total sitting time, each state, a counter for determining the current state.

**For calculation:** Calculations to determine the median of shoulder positions, the median of items, and the 								percentage representation of each state.



```
#===============================================#
#                Global Variable                #
#===============================================#

# YOLOv8n-pose Model Load(detection of human behavior)
pose_model = YOLO('yolov8m-pose.pt')
# YOLOv8m Model Load (for mobile phone detection)
phone_model = YOLO('yolov8m.pt')

# Open video stream
cap = cv2.VideoCapture(1)
# Load video# Load video
# cap = cv2.VideoCapture('4time.mov') 

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize all flags to False
flags = {f"flag_{i}": False for i in range(1, 16)}

#State calculation count
sit_time = 0
study_cnt = 0
phone_cnt = 0
sleep_cnt = 0

# Initialize each joint coordinate value
right_shoulder = (0, 0)
left_shoulder = (0, 0)
right_elbow = (0, 0)
left_elbow = (0, 0)
right_wrist = (0, 0)
left_wrist = (0, 0)

#Object detection and calculation of coordinate values
phone_detected = False
keyboard_detected = False
phone_x = 0
phone_y = 0
keyboard_x = 0
keyboard_y = 0


#Pause flag when you go to the bathroom or have an urgent need, the flag showing the last result
paused = False

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (1080, 720))
```



### 5. Definition Function

#### 5-1. Processing

**Detect phone and keyboard.**

- Detect the keyboard and phone based on flag conditions and receive their coordinate values and Calculate the median values.

```
def detect_objects(resized_frame):
    #Object detection and calculation of coordinate values
    global phone_detected, keyboard_detected, phone_x, phone_y, keyboard_x, keyboard_y

    object_results = phone_model(resized_frame)

    phone_detected = False
    keyboard_detected = False
    phone_x = 0
    phone_y = 0
    keyboard_x = 0
    keyboard_y = 0

    for detection in object_results[0].boxes:
        if phone_detected and keyboard_detected:
            break

        class_id = int(detection.cls[0])
        confidence = detection.conf[0]

        #Only one phone detected
        if class_id == 67 and confidence > 0.3 and not phone_detected:
            phone_detected = True
            box = detection.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            phone_x = (x1 + x2) / 2
            phone_y = (y1 + y2) / 2
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(resized_frame, 'Phone', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        #@Only one keyboard detected
        if class_id == 66 and confidence > 0.5 and not keyboard_detected:
            keyboard_detected = True
            box = detection.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            keyboard_x = (x1 + x2) / 2
            keyboard_y = (y1 + y2) / 2
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized_frame, 'Keyboard', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
```



**Extract both shoulders, elbows, and wrists from the 17 joints.**

- Extract both shoulders, elbows, and wrists from the 17 joints and obtain the coordinates for each frame from the extracted joints to perform calculations.

```
def detect_pose(resized_frame):
    global right_shoulder, left_shoulder, right_elbow, left_elbow, right_wrist, left_wrist

    pose_results = pose_model(resized_frame)

    for result in pose_results:
        if hasattr(result, 'keypoints'):
            keypoints = result.keypoints.data.cpu().numpy()

        #Getting each joint sensing tail joint coordinates
        if keypoints is not None:
            for person_id, person in enumerate(keypoints):
                for joint_id, joint in enumerate(person):
                    if len(joint) >= 3:
                        x, y, conf = joint
                        if conf > 0.8:
                            if joint_id == 5:
                                right_shoulder = (int(x), int(y))
                            elif joint_id == 6:
                                left_shoulder = (int(x), int(y))
                            elif joint_id == 7:
                                right_elbow = (int(x), int(y))
                            elif joint_id == 8:
                                left_elbow = (int(x), int(y))
                            elif joint_id == 9:
                                right_wrist = (int(x), int(y))
                            elif joint_id == 10:
                                left_wrist = (int(x), int(y))
```



**Connect each coordinate and prevent outlier values.**

- This is a function that can draw a skeleton model for a joint.
- This is a function also, Although the accuracy of detecting a person has improved, there are still outlier values. If a coordinate significantly differs from other coordinates, it will be excluded.

```
def display_keypoints(resized_frame):
    shoulder_center = None

    #Calculate the median shoulder
    if right_shoulder and left_shoulder:
        shoulder_center = calculate_center(right_shoulder, left_shoulder)


    #Remove Bouncing Values
    if right_shoulder and abs(right_shoulder[0] - right_elbow[0]) <= 300:
        cv2.circle(resized_frame, right_shoulder, 5, (0, 0, 255), -1)
    if left_shoulder and abs(left_shoulder[0] - left_elbow[0]) <= 300:
        cv2.circle(resized_frame, left_shoulder, 5, (0, 0, 255), -1)
    if right_elbow and abs(right_elbow[0] - right_wrist[0]) <= 300:
        cv2.circle(resized_frame, right_elbow, 5, (255, 0, 0), -1)
    if left_elbow and abs(left_elbow[0] - left_wrist[0]) <= 300:
        cv2.circle(resized_frame, left_elbow, 5, (255, 0, 0), -1)
    if right_wrist:
        cv2.circle(resized_frame, right_wrist, 5, (0, 255, 255), -1)
    if left_wrist:
        cv2.circle(resized_frame, left_wrist, 5, (0, 255, 255), -1)
    if shoulder_center:
        cv2.circle(resized_frame, shoulder_center, 5, (0, 255, 0), -1)

    #From the center of the shoulder to both shoulders,/ right shoulder, elbow, wrist order,/ left shoulder, elbow, wrist order
    if shoulder_center and right_shoulder:
        cv2.line(resized_frame, shoulder_center, right_shoulder, (255, 255, 0), 2)
    if shoulder_center and left_shoulder:
        cv2.line(resized_frame, shoulder_center, left_shoulder, (255, 255, 0), 2)
    if right_shoulder and right_elbow:
        cv2.line(resized_frame, right_shoulder, right_elbow, (255, 255, 0), 2)
    if right_elbow and right_wrist:
        cv2.line(resized_frame, right_elbow, right_wrist, (255, 255, 0), 2)
    if left_shoulder and left_elbow:
        cv2.line(resized_frame, left_shoulder, left_elbow, (255, 255, 0), 2)
    if left_elbow and left_wrist:
        cv2.line(resized_frame, left_elbow, left_wrist, (255, 255, 0), 2)
```



**check_flags**

- Fifteen flags have been created to detect the states of studying, using a phone, and sleeping. These flags are combined to define each specific state.

```
def check_flags(left_wrist, left_elbow, left_shoulder, right_wrist, right_elbow, right_shoulder, phone_detected, phone_x):
    flags = {f"flag_{i}": False for i in range(1, 16)}

    if right_wrist[1] > right_elbow[1] > right_shoulder[1]:
        flags["flag_1"] = True
    if left_wrist[1] > left_elbow[1] > left_shoulder[1]:
        flags["flag_2"] = True
    if right_elbow[1] > right_wrist[1] > right_shoulder[1]:
        flags["flag_3"] = True
    if left_elbow[1] > left_wrist[1] > left_shoulder[1]:
        flags["flag_4"] = True
    if (right_wrist[1] - right_shoulder[1]) > 40:
        flags["flag_5"] = True
    if (left_wrist[1] - left_shoulder[1]) > 40:
        flags["flag_6"] = True
    if right_elbow[1] > right_shoulder[1] > right_wrist[1]:
        flags["flag_7"] = True
    if left_elbow[1] > left_shoulder[1] > left_wrist[1]:
        flags["flag_8"] = True
    if left_wrist[0] > right_wrist[0]:
        flags["flag_9"] = True
    if abs(right_wrist[0] - left_wrist[0]) >= 250:
        flags["flag_10"] = True
    if abs(right_wrist[0] - left_wrist[0]) <= 250:
        flags["flag_11"] = True
    if left_wrist[0] < right_wrist[0]:
        flags["flag_12"] = True
    if abs(phone_x - right_wrist[0]) < 200:
        flags["flag_13"] = True
    if abs(phone_x - left_wrist[0]) < 200:
        flags["flag_14"] = True
    if abs(keyboard_y - left_wrist[1]) < 100:
        flags["flag_15"] = True

    return flags
```



**check_and_display_state**

- After checking each state, output which state it is.

- ### Studying

  - The left hand is positioned to the left of the right hand.
  - Both hands are higher than the shoulders, with a wide distance between them.
  - The right hand is higher than the shoulder while the left hand is lower than the shoulder, or vice versa.
  - The left hand is higher than shoulder height, and the right hand is lower than shoulder height, or vice versa.

  ### Using Smart Phone

  - When a phone is detected:
    - Both hands are higher than the shoulders, and their positions are close to the phone's position.
    - One hand is higher than the shoulder, and the other hand is lower than the shoulder, with the hand close to the phone's position.
    - The right hand is higher than the shoulder and the right elbow is below the shoulder.
    - The left hand is higher than the shoulder and the left elbow is below the shoulder.

  ### Sleeping

  - The left hand is positioned to the right of the right hand.

```
def check_and_display_state(resized_frame):
    global sit_time, study_cnt, phone_cnt, sleep_cnt

    state = ""
    sit_time += 1

    flags = check_flags(left_wrist, left_elbow, left_shoulder, right_wrist, right_elbow, right_shoulder, phone_detected, phone_x)

    if flags["flag_9"]:
        state = "Sleeping"

    elif flags["flag_12"]:
        if (flags["flag_1"] and flags["flag_2"] and flags["flag_10"]) or \
                (flags["flag_1"] and (flags["flag_4"] or flags["flag_8"])) or \
                (flags["flag_2"] and (flags["flag_3"] or flags["flag_7"])):
            state = "Studying"

    if flags["flag_15"]:
        state = "Studying"

    if phone_detected:
        if (flags["flag_1"] and flags["flag_2"] and (flags["flag_13"] or flags["flag_14"])) or \
                (flags["flag_3"] and flags["flag_4"] and (flags["flag_13"] or flags["flag_14"])) or \
                (flags["flag_3"] and flags["flag_5"]) or \
                (flags["flag_4"] and flags["flag_6"]):
            state = "Using Smart Phone"

    if state == "Sleeping":
        sleep_cnt += 1
    elif state == "Studying":
        study_cnt += 1
    elif state == "Using Smart Phone":
        phone_cnt += 1

    if state:
        cv2.putText(resized_frame, f'State: {state}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
```



**Save Frame Results for Status**

- Store the state for each frame for algorithm analysis.

```
def record_frame_state(frame_count, state):
    with open('study.txt', 'a') as f:
        f.write(f'{frame_count}, {state}\n')
```





#### 5-2. Show Result

**Calculate and print the state ratios**

- Show the counted values of the current state as a percentage of the total count.

```
def calculate_and_display_times(resized_frame):
    study_time = (study_cnt / sit_time) * 100 if sit_time > 0 else 0
    phone_time = (phone_cnt / sit_time) * 100 if sit_time > 0 else 0
    sleep_time = (sleep_cnt / sit_time) * 100 if sit_time > 0 else 0

    cv2.rectangle(resized_frame, (140, 210), (940, 510), (0, 0, 0), -1)
    cv2.putText(resized_frame, f'Studying: {study_time:.0f}%', (350, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(resized_frame, f'Using Smart Phone: {phone_time:.0f}%', (350, 410), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(resized_frame, f'Sleeping: {sleep_time:.0f}%', (350, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
```



### 7. Main Code

**Model Interpreter Definition**

- Load the model (Write it directly underneath the code where the library is configured.)

  ```
  # Load YOLOv8n-pose model
  pose_model = YOLO('yolov8n-pose.pt')
  # Load YOLOv8m model (for phone detection)
  phone_model = YOLO('yolov8m.pt')
  ```

  ![image](https://github.com/12-dimension-cat/neko/assets/144550430/9be57a73-a786-4249-81e6-e3336c31a2d5)
  
  <center>  Figure 9. Load the model code



**Load webcam or video file**

- Also Write it directly underneath the code where the library is configured.

```
# Open video stream
cap = cv2.VideoCapture(1)
# Load saved video
# cap = cv2.VideoCapture('4time.mov')
```

![image](https://github.com/12-dimension-cat/neko/assets/144550430/a3135e4d-59b8-44eb-b1c3-6ce905e95948)

<center>  Figure 10. Load webcam or video file code



**Set video screen size**

```
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

![image](https://github.com/12-dimension-cat/neko/assets/144550430/91493734-9620-4919-ba97-7d426cb9f8e9)

<center>  Figure 11. Set video screen size code



**Set the codec for saving the result video.**

```
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (1080, 720))
```

<img src="https://github.com/12-dimension-cat/neko/assets/144550430/a3032188-2ee6-4af4-aeee-717e6ee042cc" alt="image" style="zoom: 67%;" />

<center>  Figure 12. Set the codec for saving the result video code.

**Main Code**

- Write defined function definitions at the bottom.

```
def main():
    global paused
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if not paused:
            resized_frame = cv2.resize(frame, (1080, 720))
            detect_objects(resized_frame)
            detect_pose(resized_frame)
            display_keypoints(resized_frame)
            check_and_display_state(resized_frame)
            #record_frame_state(frame_count, "Studying" ) ## or other sate
            frame_count += 1

            # Save the frame to the video
            out.write(resized_frame)

        cv2.imshow("YOLOv8n-pose Skeleton", resized_frame)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break
        elif key == 13:  # Enter key
            paused = not paused
            if paused:
                calculate_and_display_times(resized_frame)

    # Release video objects
    out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```



## V. Result



### 1. Studying

![image](https://github.com/12-dimension-cat/neko/assets/144550430/f4d2554c-997b-4d23-b14a-7c38f03cd628)

<center>  Figure 13. Studying State Result



### 2. Using phone

![image](https://github.com/12-dimension-cat/neko/assets/144550430/4eedbd36-7346-4dd7-afdc-aa09ec09f2c7)

<center>  Figure 14. Using phone State Result



### 3. Sleeping

![image](https://github.com/12-dimension-cat/neko/assets/144550430/4ced21bb-58ff-4772-bcb9-7e59836e0fed)

<center>  Figure 15. Sleeping State Result





### 4. Total each state Result

<img src="https://github.com/12-dimension-cat/neko/assets/144550430/e047d46d-4d05-483f-8fab-c54e8f678e44" alt="image" style="zoom:50%;" />

<center>  Figure 16. Results of each state ratio





### 5. Results of Other Human

<img src="https://github.com/12-dimension-cat/neko/assets/144550430/3f0424d6-088f-4e74-96e9-2efbf63b84d2" alt="image" style="zoom:33%;" />

<img src="https://github.com/12-dimension-cat/neko/assets/144550430/5930d2e6-ae61-4ddf-b7c4-bcaee29fd186" alt="image" style="zoom:33%;" />

<img src="https://github.com/12-dimension-cat/neko/assets/144550430/79d11729-6631-4b7a-aad3-8124efb9563b" alt="image" style="zoom:33%;" />

<center>  Figure 17. State results of Other Human

### 6. Result Video

[Demo Video](https://youtu.be/-X37Pc_AxDw)

[Other Human Result](https://youtu.be/vrD-Mg7xqZw)



## VI. Evaluation

Since we used the pre-trained model, we analyzed the algorithm we implemented, not the analysis of the model itself. It was not possible to evaluate all states, so the states were defined, and evaluations were conducted using videos of those states. Since one frame is approximately 2ms, values measured incorrectly in two or more frames should be marked as error values.



- **Studying State **

  In the case of evaluating the algorithm for a studying state, the current state was defined as studying, and a video of the studying state was taken for evaluation. TP calculates the number of times the model correctly predicts values that exist in the actual data. FP calculates the number of times the model incorrectly predicts a value as positive when it does not exist in the actual data. FN calculates the number of times the model incorrectly predicts a value as negative when it exists in the actual data. TN refers to the number of times the model correctly predicts values as negative among those that are actually negative.

  ![image](https://github.com/12-dimension-cat/neko/assets/144550430/5552fbb5-ff21-4443-82ae-3c1e67e4e501)

  <center>  Figure 18. Binary Classification of Studying state

  **- Accuracy: 99.83%**

  **- Precision: 99.99%**

  **- Recall: 99.84%**

  

- **Using phone State **

  In the case of evaluating the algorithm for a Using phone State, the current state was defined as Using phone, and a video of the studying state was taken for evaluation. TP calculates the number of times the model correctly predicts values that exist in the actual data. FP calculates the number of times the model incorrectly predicts a value as positive when it does not exist in the actual data. FN calculates the number of times the model incorrectly predicts a value as negative when it exists in the actual data. TN refers to the number of times the model correctly predicts values as negative among those that are actually negative.

  ![image](https://github.com/12-dimension-cat/neko/assets/144550430/2a977066-b096-44fa-8dda-1fbdd2326213)

  <center>  Figure 19. Binary Classification of Using phone state

  **- Accuracy: 98.12%**

  **- Precision: 98.12%**

  **- Recall: 100.0%**

  

  **Sleeping State **

  In the case of evaluating the algorithm for a Sleeping  State, the current state was defined as Sleeping, and a video of the studying state was taken for evaluation. TP calculates the number of times the model correctly predicts values that exist in the actual data. FP calculates the number of times the model incorrectly predicts a value as positive when it does not exist in the actual data. FN calculates the number of times the model incorrectly predicts a value as negative when it exists in the actual data. TN refers to the number of times the model correctly predicts values as negative among those that are actually negative.

  ![image](https://github.com/12-dimension-cat/neko/assets/144550430/2a977066-b096-44fa-8dda-1fbdd2326213)

  <center>  Figure 20. Binary Classification of Sleeping  state

  **- Accuracy: 99.99%**
  
  **- Precision: 99.99%**
  
  **- Recall: 100.0%**



**Total State **

The result is the sum of TP, FP, FN, and TN of the above three evaluations.

![image](https://github.com/12-dimension-cat/neko/assets/144550430/d154ad4c-b3f3-4569-b578-d721df0053dd)

<center>  Figure 21. Binary Classification of Total  state

**- Accuracy: 99.08%**

**- Precision: 99.99%**

**- Recall: 99.09%**



Looking at the results, since all values are 99%, excluding outliers, it can be seen that the algorithm works well. However, human behavior is complex and varies from person to person, so the state of studying can differ for each individual. Therefore, if we analyze the person's behavior patterns and modify the flags accordingly, the feedback on the studying state could be more accurate.



## VII. Reference

- [Pose - Ultralytics YOLO Docs](https://docs.ultralytics.com/tasks/pose/)[Pose - Ultralytics YOLO Docs](https://docs.ultralytics.com/tasks/pose/)
- [Installation Guide - Y.K.Kim Deep-Learning-Image-Processing Installation Guide](https://ykkim.gitbook.io/dlip/installation-guide/installation-guide-for-deep-learning)



## VIII. Appendix

```python
import cv2
from ultralytics import YOLO
import numpy as np

# YOLOv8n-pose Model Load(detection of human behavior)
pose_model = YOLO('yolov8m-pose.pt')
# YOLOv8m Model Load (for mobile phone detection)
phone_model = YOLO('yolov8m.pt')

# Open video stream
cap = cv2.VideoCapture(1)
# Load video# Load video
# cap = cv2.VideoCapture('4time.mov') 

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize flags and counters
flags = {f"flag_{i}": False for i in range(1, 16)}

#State calculation count
sit_time = 0
study_cnt = 0
phone_cnt = 0
sleep_cnt = 0

# Initialize each joint coordinate value
right_shoulder = (0, 0)
left_shoulder = (0, 0)
right_elbow = (0, 0)
left_elbow = (0, 0)
right_wrist = (0, 0)
left_wrist = (0, 0)

#Pause flag when you go to the bathroom or have an urgent need, the flag showing the last result
paused = False

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (1080, 720))

def calculate_and_display_times(resized_frame):
    study_time = (study_cnt / sit_time) * 100 if sit_time > 0 else 0
    phone_time = (phone_cnt / sit_time) * 100 if sit_time > 0 else 0
    sleep_time = (sleep_cnt / sit_time) * 100 if sit_time > 0 else 0

    cv2.rectangle(resized_frame, (140, 210), (940, 510), (0, 0, 0), -1)
    cv2.putText(resized_frame, f'Studying: {study_time:.0f}%', (350, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(resized_frame, f'Using Smart Phone: {phone_time:.0f}%', (350, 410), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(resized_frame, f'Sleeping: {sleep_time:.0f}%', (350, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def calculate_center(coord1, coord2):
    return ((coord1[0] + coord2[0]) // 2, (coord1[1] + coord2[1]) // 2)

def check_flags(left_wrist, left_elbow, left_shoulder, right_wrist, right_elbow, right_shoulder, phone_detected, phone_x):
    flags = {f"flag_{i}": False for i in range(1, 16)}

    if right_wrist[1] > right_elbow[1] > right_shoulder[1]:
        flags["flag_1"] = True
    if left_wrist[1] > left_elbow[1] > left_shoulder[1]:
        flags["flag_2"] = True
    if right_elbow[1] > right_wrist[1] > right_shoulder[1]:
        flags["flag_3"] = True
    if left_elbow[1] > left_wrist[1] > left_shoulder[1]:
        flags["flag_4"] = True
    if (right_wrist[1] - right_shoulder[1]) > 40:
        flags["flag_5"] = True
    if (left_wrist[1] - left_shoulder[1]) > 40:
        flags["flag_6"] = True
    if right_elbow[1] > right_shoulder[1] > right_wrist[1]:
        flags["flag_7"] = True
    if left_elbow[1] > left_shoulder[1] > left_wrist[1]:
        flags["flag_8"] = True
    if left_wrist[0] > right_wrist[0]:
        flags["flag_9"] = True
    if abs(right_wrist[0] - left_wrist[0]) >= 250:
        flags["flag_10"] = True
    if abs(right_wrist[0] - left_wrist[0]) <= 250:
        flags["flag_11"] = True
    if left_wrist[0] < right_wrist[0]:
        flags["flag_12"] = True
    if abs(phone_x - right_wrist[0]) < 200:
        flags["flag_13"] = True
    if abs(phone_x - left_wrist[0]) < 200:
        flags["flag_14"] = True
    if abs(keyboard_y - left_wrist[1]) < 100:
        flags["flag_15"] = True

    return flags

def detect_objects(resized_frame):
    #Object detection and calculation of coordinate values
    global phone_detected, keyboard_detected, phone_x, phone_y, keyboard_x, keyboard_y

    object_results = phone_model(resized_frame)

    phone_detected = False
    keyboard_detected = False
    phone_x = 0
    phone_y = 0
    keyboard_x = 0
    keyboard_y = 0

    for detection in object_results[0].boxes:
        if phone_detected and keyboard_detected:
            break

        class_id = int(detection.cls[0])
        confidence = detection.conf[0]

        #Only one phone detected
        if class_id == 67 and confidence > 0.3 and not phone_detected:
            phone_detected = True
            box = detection.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            phone_x = (x1 + x2) / 2
            phone_y = (y1 + y2) / 2
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(resized_frame, 'Phone', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        #@Only one keyboard detected
        if class_id == 66 and confidence > 0.5 and not keyboard_detected:
            keyboard_detected = True
            box = detection.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            keyboard_x = (x1 + x2) / 2
            keyboard_y = (y1 + y2) / 2
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized_frame, 'Keyboard', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def detect_pose(resized_frame):
    global right_shoulder, left_shoulder, right_elbow, left_elbow, right_wrist, left_wrist

    pose_results = pose_model(resized_frame)

    for result in pose_results:
        if hasattr(result, 'keypoints'):
            keypoints = result.keypoints.data.cpu().numpy()

        #Getting each joint sensing tail joint coordinates
        if keypoints is not None:
            for person_id, person in enumerate(keypoints):
                for joint_id, joint in enumerate(person):
                    if len(joint) >= 3:
                        x, y, conf = joint
                        if conf > 0.8:
                            if joint_id == 5:
                                right_shoulder = (int(x), int(y))
                            elif joint_id == 6:
                                left_shoulder = (int(x), int(y))
                            elif joint_id == 7:
                                right_elbow = (int(x), int(y))
                            elif joint_id == 8:
                                left_elbow = (int(x), int(y))
                            elif joint_id == 9:
                                right_wrist = (int(x), int(y))
                            elif joint_id == 10:
                                left_wrist = (int(x), int(y))

def display_keypoints(resized_frame):
    shoulder_center = None

    #Calculate the median shoulder
    if right_shoulder and left_shoulder:
        shoulder_center = calculate_center(right_shoulder, left_shoulder)


    #Remove Bouncing Values
    if right_shoulder and abs(right_shoulder[0] - right_elbow[0]) <= 300:
        cv2.circle(resized_frame, right_shoulder, 5, (0, 0, 255), -1)
    if left_shoulder and abs(left_shoulder[0] - left_elbow[0]) <= 300:
        cv2.circle(resized_frame, left_shoulder, 5, (0, 0, 255), -1)
    if right_elbow and abs(right_elbow[0] - right_wrist[0]) <= 300:
        cv2.circle(resized_frame, right_elbow, 5, (255, 0, 0), -1)
    if left_elbow and abs(left_elbow[0] - left_wrist[0]) <= 300:
        cv2.circle(resized_frame, left_elbow, 5, (255, 0, 0), -1)
    if right_wrist:
        cv2.circle(resized_frame, right_wrist, 5, (0, 255, 255), -1)
    if left_wrist:
        cv2.circle(resized_frame, left_wrist, 5, (0, 255, 255), -1)
    if shoulder_center:
        cv2.circle(resized_frame, shoulder_center, 5, (0, 255, 0), -1)

    #From the center of the shoulder to both shoulders,/ right shoulder, elbow, wrist order,/ left shoulder, elbow, wrist order
    if shoulder_center and right_shoulder:
        cv2.line(resized_frame, shoulder_center, right_shoulder, (255, 255, 0), 2)
    if shoulder_center and left_shoulder:
        cv2.line(resized_frame, shoulder_center, left_shoulder, (255, 255, 0), 2)
    if right_shoulder and right_elbow:
        cv2.line(resized_frame, right_shoulder, right_elbow, (255, 255, 0), 2)
    if right_elbow and right_wrist:
        cv2.line(resized_frame, right_elbow, right_wrist, (255, 255, 0), 2)
    if left_shoulder and left_elbow:
        cv2.line(resized_frame, left_shoulder, left_elbow, (255, 255, 0), 2)
    if left_elbow and left_wrist:
        cv2.line(resized_frame, left_elbow, left_wrist, (255, 255, 0), 2)


def check_and_display_state(resized_frame):
    global sit_time, study_cnt, phone_cnt, sleep_cnt

    state = ""
    sit_time += 1

    flags = check_flags(left_wrist, left_elbow, left_shoulder, right_wrist, right_elbow, right_shoulder, phone_detected, phone_x)

    if flags["flag_9"]:
        state = "Sleeping"

    elif flags["flag_12"]:
        if (flags["flag_1"] and flags["flag_2"] and flags["flag_10"]) or \
                (flags["flag_1"] and (flags["flag_4"] or flags["flag_8"])) or \
                (flags["flag_2"] and (flags["flag_3"] or flags["flag_7"])):
            state = "Studying"

    if flags["flag_15"]:
        state = "Studying"

    if phone_detected:
        if (flags["flag_1"] and flags["flag_2"] and (flags["flag_13"] or flags["flag_14"])) or \
                (flags["flag_3"] and flags["flag_4"] and (flags["flag_13"] or flags["flag_14"])) or \
                (flags["flag_3"] and flags["flag_5"]) or \
                (flags["flag_4"] and flags["flag_6"]):
            state = "Using Smart Phone"

    if state == "Sleeping":
        sleep_cnt += 1
    elif state == "Studying":
        study_cnt += 1
    elif state == "Using Smart Phone":
        phone_cnt += 1

    if state:
        cv2.putText(resized_frame, f'State: {state}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def record_frame_state(frame_count, state):
    with open('phone.txt', 'a') as f:
        f.write(f'{frame_count}, {state}\n')

def main():
    global paused
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if not paused:
            resized_frame = cv2.resize(frame, (1080, 720))
            detect_objects(resized_frame)
            detect_pose(resized_frame)
            display_keypoints(resized_frame)
            check_and_display_state(resized_frame)
            #record_frame_state(frame_count, "Studying" ) ## or other sate
            frame_count += 1

            # Save the frame to the video
            out.write(resized_frame)

        cv2.imshow("YOLOv8n-pose Skeleton", resized_frame)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break
        elif key == 13:  # Enter key
            paused = not paused
            if paused:
                calculate_and_display_times(resized_frame)

    # Release video objects
    out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

# Anaconda Installation and Environment Setup Guide

## 1. Installing Anaconda and Creating an Environment

### Install Anaconda
If Anaconda is not installed, download the installer from the [Anaconda download page](https://www.anaconda.com/products/distribution) and install it.

### Create a Virtual Environment
Open the Anaconda prompt and enter the following command to create a virtual environment:
```bash
conda create -n myenv python=3.9.18
```
`myenv` is the name of the virtual environment. You can change it to any desired name.

## 2. Activate the Virtual Environment
Activate the created virtual environment.
```bash
conda activate myenv
```

## 3. Install Required Packages

### Install numpy
```bash
pip install numpy==1.26.4
```

### Install OpenCV
```bash
pip install opencv-python==4.10.0
```

## 4. Install PyTorch and CUDA

### Check Graphics Card and CUDA Compatibility

#### Check NVIDIA Driver Installation and Version
To check if CUDA is available, an NVIDIA graphics card must be installed. Use the following command to check:
```bash
nvidia-smi
```
This command shows the currently installed NVIDIA driver and CUDA version. If the `nvidia-smi` command does not work, the NVIDIA driver may not be installed or an NVIDIA graphics card may not be present.

#### Check CUDA Compatibility
Visit the [CUDA compatibility page](https://developer.nvidia.com/cuda-gpus) to check which CUDA version your graphics card supports.

### Install PyTorch and CUDA
To install PyTorch, refer to the [PyTorch official website](https://pytorch.org/get-started/locally/) Get Started page for the appropriate command. For example, to install PyTorch 2.1.2 with CUDA 11.8:
```bash
pip install torch==2.1.2+cu118 torchvision==0.15.2+cu118 torchaudio==2.1.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

## 5. Install YOLOv8 and YOLOv8-pose
YOLOv8 and YOLOv8-pose can be installed from the Ultralytics YOLOv8 repository. First, clone it via git and then install it.
```bash
# Install Yolov8
pip install ultralytics

# Install Yolov8-pose (yolov8-pose has extended features from yolov8, so additional packages may be required.)
pip install -U ultralytics[pose]
```

## 6. Verify Installation
To ensure all packages are installed correctly, use the following commands to print the version of each package.
```bash
python -c "import numpy as np; print(np.__version__)"
python -c "import cv2; print(cv2.__version__)"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import ultralytics; print(ultralytics.__version__)"
```

## 7. Start Working in the Virtual Environment
Perform the required tasks in the activated virtual environment. Install additional packages or change settings as needed.
