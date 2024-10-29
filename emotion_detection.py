import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO('yolov8_emotion_best_3.pt')  # Replace with the path to your saved model

# Open the webcam (0 is typically the built-in webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define your emotion labels (replace with your dataset labels)
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']  # Replace with actual emotion labels

# Start webcam emotion detection
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Use the model to predict emotions on the current frame
    results = model.predict(source=frame, save=False)

    # Loop through the results and draw bounding boxes + emotions
    for result in results:
        for box in result.boxes:
            # Get bounding box coordinates and class index
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            emotion = class_labels[cls]

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Emotion Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
