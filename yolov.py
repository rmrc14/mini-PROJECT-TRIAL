import cv2
import numpy as np
from PIL import Image
from sort import Sort

# Load YOLOv3 and Deep SORT models
# Replace 'path/to/yolov3.weights' with the path to your YOLOv3 weights file
# Replace 'path/to/deep_sort_model' with the path to your Deep SORT model file
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
tracker = Sort('path/to/deep_sort_model')

# Define vehicle classes
classNames = ['car', 'truck', 'bus', 'motorbike', 'bicycle']

# Define line coordinates for counting vehicles
line_coordinates = [(300, 500), (900, 500)]

# Load the video
video_path = 'test1.mp4'
cap = cv2.VideoCapture(video_path)

# Define the output video path and parameters
output_path = 'output.avi'
output_fps = cap.get(cv2.CAP_PROP_FPS)
output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object to save the processed video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))

# Initialize variables for vehicle counting
vehicle_count_in = 0
vehicle_count_out = 0

while True:
    # Read the next frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection with YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    # Process detection results
    boxes = []
    confidences = []
    class_ids = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classNames[class_id] in ['car', 'truck', 'bus', 'motorbike', 'bicycle']:
                center_x = int(detection[0] * output_width)
                center_y = int(detection[1] * output_height)
                width = int(detection[2] * output_width)
                height = int(detection[3] * output_height)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform vehicle tracking with Deep SORT
    detections = np.array(boxes + confidences).reshape(-1, 5)
    tracked_objects = tracker.update(detections)

    # Draw detection results and track IDs on the frame
    for detection in tracked_objects:
        x, y, w, h, track_id = detection.astype(np.int32)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(track_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Check if the vehicle crosses the counting line
        center_x = x + w // 2
        center_y = y + h // 2
        if center_x > line_coordinates[0][0] and center_x < line_coordinates[1][0] and \
                abs(center_y - line_coordinates[0][1]) < 10:
            vehicle_count_in += 1
        elif center_x > line_coordinates[0][0] and center_x < line_coordinates[1][0] and \
                abs(center_y - line_coordinates[0][1]) > 10:
            vehicle_count_out += 1

    # Draw the counting line on the frame
    cv2.line(frame, line_coordinates[0], line_coordinates[1], (0, 0, 255), 2)

    # Display vehicle counts on the frame
    cv2.putText(frame, 'Count In: {}'.format(vehicle_count_in), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, 'Count Out: {}'.format(vehicle_count_out), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write the processed frame to the output video
    out.write(frame)

    # Display the processed frame
    cv2.imshow('Vehicle Counting', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
