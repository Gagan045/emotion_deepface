import cv2
from deepface import DeepFace

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Extract the face ROI (Region of Interest)
        face_roi = frame[y:y + h, x:x + w]

        try:
            # Emotion recognition using DeepFace
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Get the dominant emotion
            print(result[0]['emotion'])
            dominant_emotion = result[0]['emotion']
            dominant_emotion=max(zip(dominant_emotion.values(), dominant_emotion.keys()))[1]

            # Display the dominant emotion on the frame
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        except ValueError:
            # Handle the case when no face is detected
            cv2.putText(frame, "No face detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame with face rectangles and emotion
    cv2.imshow('Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
