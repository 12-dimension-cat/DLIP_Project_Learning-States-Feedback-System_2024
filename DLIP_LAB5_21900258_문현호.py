import cv2
from ultralytics import YOLO
import numpy as np

# YOLOv8n-pose Model Load(detection of human behavior)
pose_model = YOLO('yolov8m-pose.pt')
# YOLOv8m Model Load (for mobile phone detection)
phone_model = YOLO('yolov8m.pt')

# Open video stream
#cap = cv2.VideoCapture(1)
# Load video# Load video
cap = cv2.VideoCapture('DLIP_LAB5_21900258_문현호.mp4') 

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