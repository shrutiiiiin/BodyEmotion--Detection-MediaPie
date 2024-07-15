import mediapipe as mp # type: ignore
import cv2 # type: ignore

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Capture video from the default webcam (index 0)
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

# Loop to continuously capture frames and process them
    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame from the video capture
        
        # Convert the frame color from BGR to RGB for Mediapipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with the holistic model
        results = holistic.process(image_rgb)
            
        # Convert the frame color back to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
        # Draw face landmarks on the frame
        mp_drawing.draw_landmarks(image_bgr, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        
        # Draw Right hand landmark on the frame
        mp_drawing.draw_landmarks(image_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # Draw Left hand landmark on the frame
        mp_drawing.draw_landmarks(image_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Display the annotated frame
        cv2.imshow('Holistic Model Detection', image_bgr)
            
        # Check for 'q' key press to exit the loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
