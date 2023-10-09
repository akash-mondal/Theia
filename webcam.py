import cv2
import torch
import mediapipe as mp
import numpy as np
import time
import os

# Load YOLOv5 model with your specific weights
model = torch.hub.load('ultralytics/yolov5', 'custom', device='cpu', path='proctor.pt')

# Set the webcam source (usually 0 for the default webcam)
webcam_source = 0

# Initialize webcam capture
cap = cv2.VideoCapture(webcam_source)

############## PARAMETERS #######################################################

# Set these values to show/hide certain vectors of the estimation
draw_gaze = True
draw_full_axis = False
draw_headpose = True

# Gaze Score multiplier (Higher multiplier = Gaze affects head pose estimation more)
x_score_multiplier = 4
y_score_multiplier = 4

# Threshold of how close scores should be to average between frames
threshold = 0.3

# Time threshold for capturing an image (in seconds)
capture_threshold = 1.5
mobile_phone_threshold = 0.500

# Directory to save captured images
output_directory = "captured_images"

#################################################################################

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=2,
    min_detection_confidence=0.5)
face_3d = np.array([
    [0.0, 0.0, 0.0],            # Nose tip
    [0.0, -330.0, -65.0],       # Chin
    [-225.0, 170.0, -135.0],    # Left eye left corner
    [225.0, 170.0, -135.0],     # Right eye right corner
    [-150.0, -150.0, -125.0],   # Left mouth corner
    [150.0, -150.0, -125.0]     # Right mouth corner
    ], dtype=np.float64)

# Reposition left eye corner to be the origin
leye_3d = np.array(face_3d)
leye_3d[:,0] += 225
leye_3d[:,1] -= 175
leye_3d[:,2] += 135

# Reposition right eye corner to be the origin
reye_3d = np.array(face_3d)
reye_3d[:,0] -= 225
reye_3d[:,1] -= 175
reye_3d[:,2] += 135

# Gaze scores from the previous frame
last_lx, last_rx = 0, 0
last_ly, last_ry = 0, 0

start_time = None
last_violation_time = time.time()  # Initialize last_violation_time
start_mobile_time = None  # Initialize start_mobile_time

# Counter for naming peeking images
peeking_counter = 1
mobile_phone_counter = 1
# Initialize a variable to store the graph position
graph_x = 0
# Initialize the log message and create an empty log list
log_message = ""
log = []
# Initialize variables for the extra person violation
extra_person_counter = 1
extra_person_start_time = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Perform inference
    results = model(frame)

    # Get the detected objects and their coordinates
    pred = results.pred[0]

    # Reset the counters for each frame
    person_counter = 0
    cell_phone_counter = 0

    for det in pred:
        label, confidence, bbox = det[5], det[4], det[:4].cpu().numpy()
        if confidence > 0.4:  # You can adjust the confidence threshold
            x1, y1, x2, y2 = bbox.astype(int)
            class_name = model.names[int(label)]

            # Check if the detected object is a "person" or "cell phone"
            if class_name == "person":
                person_counter += 1
            elif class_name == "cell phone":
                cell_phone_counter += 1
            elif class_name == "laptop":
                cell_phone_counter += 1
            elif class_name == "suitcase":
                cell_phone_counter += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name}: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections and the counters
    cv2.putText(frame, f'Person Count: {person_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Cell Phone Count: {cell_phone_counter}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Reflect mobile phone violation in YOLOv5 object detection window
    if cell_phone_counter > 0:
        cv2.putText(frame, "Mobile Phone Detected!", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Check for extra person violation
    if person_counter > 1:
        if extra_person_start_time is None:
            extra_person_start_time = time.time()
        else:
            elapsed_extra_person_time = time.time() - extra_person_start_time
            if elapsed_extra_person_time >= 2:
                # Capture and store the image for extra person violation
                capture_time = time.strftime("%Y%m%d%H%M%S")
                image_filename = f"{output_directory}/ViolationExtraPerson{extra_person_counter}.png"
                cv2.imwrite(image_filename, frame)
                print(f"Captured image: {image_filename}")
                extra_person_counter += 1
                extra_person_start_time = None

                # Reflect extra person violation in YOLOv5 object detection window
                cv2.putText(frame, "Extra Person Violation Detected!", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, log_message, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.imshow('YOLOv5 Object Detection', frame)

    success, img= cap.read()

    # Flip + convert img from BGR to RGB
    img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    img.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(img)
    img.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    (img_h, img_w, img_c) = img.shape
    face_2d = []

    if not results.multi_face_landmarks:
        # Reset the timer when no face is detected
        start_time = None
        continue 

    for face_landmarks in results.multi_face_landmarks:
        face_2d = []
        for idx, lm in enumerate(face_landmarks.landmark):
            # Convert landmark x and y to pixel coordinates
            x, y = int(lm.x * img_w), int(lm.y * img_h)

            # Add the 2D coordinates to an array
            face_2d.append((x, y))
        
        # Get relevant landmarks for head pose estimation
        face_2d_head = np.array([
            face_2d[1],      # Nose
            face_2d[199],    # Chin
            face_2d[33],     # Left eye left corner
            face_2d[263],    # Right eye right corner
            face_2d[61],     # Left mouth corner
            face_2d[291]     # Right mouth corner
        ], dtype=np.float64)

        face_2d = np.asarray(face_2d)

        # Calculate left x gaze score
        if (face_2d[243,0] - face_2d[130,0]) != 0:
            lx_score = (face_2d[468,0] - face_2d[130,0]) / (face_2d[243,0] - face_2d[130,0])
            if abs(lx_score - last_lx) < threshold:
                lx_score = (lx_score + last_lx) / 2
            last_lx = lx_score

        # Calculate left y gaze score
        if (face_2d[23,1] - face_2d[27,1]) != 0:
            ly_score = (face_2d[468,1] - face_2d[27,1]) / (face_2d[23,1] - face_2d[27,1])
            if abs(ly_score - last_ly) < threshold:
                ly_score = (ly_score + last_ly) / 2
            last_ly = ly_score

        # Calculate right x gaze score
        if (face_2d[359,0] - face_2d[463,0]) != 0:
            rx_score = (face_2d[473,0] - face_2d[463,0]) / (face_2d[359,0] - face_2d[463,0])
            if abs(rx_score - last_rx) < threshold:
                rx_score = (rx_score + last_rx) / 2
            last_rx = rx_score

        # Calculate right y gaze score
        if (face_2d[253,1] - face_2d[257,1]) != 0:
            ry_score = (face_2d[473,1] - face_2d[257,1]) / (face_2d[253,1] - face_2d[257,1])
            if abs(ry_score - last_ry) < threshold:
                ry_score = (ry_score + last_ry) / 2
            last_ry = ry_score

        # The camera matrix
        focal_length = 1 * img_w
        cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                [0, focal_length, img_w / 2],
                                [0, 0, 1]])

        # Distortion coefficients 
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        _, l_rvec, l_tvec = cv2.solvePnP(leye_3d, face_2d_head, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        _, r_rvec, r_tvec = cv2.solvePnP(reye_3d, face_2d_head, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # Get rotational matrix from rotational vector
        l_rmat, _ = cv2.Rodrigues(l_rvec)
        r_rmat, _ = cv2.Rodrigues(r_rvec)

        # [0] changes pitch
        # [1] changes roll
        # [2] changes yaw
        # +1 changes ~45 degrees (pitch down, roll tilts left (counterclockwise), yaw spins left (counterclockwise))

        # Adjust headpose vector with gaze score
        l_gaze_rvec = np.array(l_rvec)
        l_gaze_rvec[2][0] -= (lx_score-.5) * x_score_multiplier
        l_gaze_rvec[0][0] += (ly_score-.5) * y_score_multiplier

        r_gaze_rvec = np.array(r_rvec)
        r_gaze_rvec[2][0] -= (rx_score-.5) * x_score_multiplier
        r_gaze_rvec[0][0] += (ry_score-.5) * y_score_multiplier

        # --- Projection ---

        # Get left eye corner as integer
        l_corner = face_2d_head[2].astype(np.int32)

        # Project axis of rotation for left eye
        axis = np.float32([[-100, 0, 0], [0, 100, 0], [0, 0, 300]]).reshape(-1, 3)
        l_axis, _ = cv2.projectPoints(axis, l_rvec, l_tvec, cam_matrix, dist_coeffs)
        l_gaze_axis, _ = cv2.projectPoints(axis, l_gaze_rvec, l_tvec, cam_matrix, dist_coeffs)

        # Draw axis of rotation for left eye
        if draw_headpose:
            if draw_full_axis:
                cv2.line(img, l_corner, tuple(np.ravel(l_axis[0]).astype(np.int32)), (200,200,0), 3)
                cv2.line(img, l_corner, tuple(np.ravel(l_axis[1]).astype(np.int32)), (0,200,0), 3)
            cv2.line(img, l_corner, tuple(np.ravel(l_axis[2]).astype(np.int32)), (0,200,200), 3)

        if draw_gaze:
            if draw_full_axis:
                cv2.line(img, l_corner, tuple(np.ravel(l_gaze_axis[0]).astype(np.int32)), (255,0,0), 3)
                cv2.line(img, l_corner, tuple(np.ravel(l_gaze_axis[1]).astype(np.int32)), (0,255,0), 3)
            cv2.line(img, l_corner, tuple(np.ravel(l_gaze_axis[2]).astype(np.int32)), (0,0,255), 3)

        # Get left eye corner as integer
        r_corner = face_2d_head[3].astype(np.int32)

        # Get left eye corner as integer
        r_axis, _ = cv2.projectPoints(axis, r_rvec, r_tvec, cam_matrix, dist_coeffs)
        r_gaze_axis, _ = cv2.projectPoints(axis, r_gaze_rvec, r_tvec, cam_matrix, dist_coeffs)

        # Draw axis of rotation for left eye
        if draw_headpose:
            if draw_full_axis:
                cv2.line(img, r_corner, tuple(np.ravel(r_axis[0]).astype(np.int32)), (200,200,0), 3)
                cv2.line(img, r_corner, tuple(np.ravel(r_axis[1]).astype(np.int32)), (0,200,0), 3)
            cv2.line(img, r_corner, tuple(np.ravel(r_axis[2]).astype(np.int32)), (0,200,200), 3)

        if draw_gaze:
            if draw_full_axis:
                cv2.line(img, r_corner, tuple(np.ravel(r_gaze_axis[0]).astype(np.int32)), (255,0,0), 3)
                cv2.line(img, r_corner, tuple(np.ravel(r_gaze_axis[1]).astype(np.int32)), (0,255,0), 3)
            cv2.line(img, r_corner, tuple(np.ravel(r_gaze_axis[2]).astype(np.int32)), (0,0,255), 3)

        # Check if the person is looking away from the camera for more than 1.5 seconds
        if abs(lx_score - 0.5) > 0.5 or abs(ly_score - 0.5) > 0.5 or abs(rx_score - 0.5) > 0.5 or abs(ry_score - 0.5) > 0.1:
            if start_time is None:
                start_time = time.time()
            else:
                elapsed_time = time.time() - start_time
                if elapsed_time >= capture_threshold:
                    # Capture and store the image for peeking violation
                    capture_time = time.strftime("%Y%m%d%H%M%S")
                    image_filename = f"{output_directory}/ViolationPeeking{peeking_counter}.png"
                    cv2.imwrite(image_filename, img)
                    print(f"Captured image: {image_filename}")
                    peeking_counter += 1
                    start_time = None
                    
                    # Check if it's time to update the graph
                    current_time = time.time()
                    if current_time - last_violation_time >= 1.0:
                        # Draw the graph
                        graph_x += 1
                        cv2.rectangle(frame, (0, 0), (graph_x * 10, 10), (0, 0, 255), -1)
                        last_violation_time = current_time
                        # Add a violation message to the log
                        violation_message = f"Violation {peeking_counter}: Looking away from the screen."
                        log_message += "\n" + violation_message  # Append new violation message
                        log.append(violation_message)
        else:
            # Reset the timer when the person is looking towards the camera
            start_time = None

        # Check if the person is using a mobile phone for more than 0.250 seconds
        if cell_phone_counter > 0:
            if start_mobile_time is None:
                start_mobile_time = time.time()
            else:
                elapsed_mobile_time = time.time() - start_mobile_time
                if elapsed_mobile_time >= mobile_phone_threshold:
                    # Capture and store the image for mobile phone violation
                    capture_time = time.strftime("%Y%m%d%H%M%S")
                    image_filename = f"{output_directory}/ViolationMobilePhone{mobile_phone_counter}.png"
                    cv2.imwrite(image_filename, img)
                    print(f"Captured image: {image_filename}")
                    mobile_phone_counter += 1
                    start_mobile_time = None
                    
                    # Reflect mobile phone violation in YOLOv5 object detection window
                    cv2.putText(frame, "Mobile Phone Violation Detected!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the log in the respective windows
    cv2.putText(frame, log_message, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.putText(img, log_message, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    cv2.imshow('Head Pose Estimation', img)


    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

