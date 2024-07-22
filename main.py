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
            
        # 1 Draw face landmarks on the frame
        mp_drawing.draw_landmarks(image_bgr, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(80,110,10), thickness=2, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80,256,121), thickness=2, circle_radius=1))
        
        # 2 Draw Right hand landmark on the frame
        mp_drawing.draw_landmarks(image_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                # BGR Format for the drawingspec color
                                  mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(240,0,0), thickness=2, circle_radius=2))
        
        # 3 Draw Left hand landmark on the frame
        mp_drawing.draw_landmarks(image_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                #   BGR Frmat for the drawingspec color right hand
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
       
        # 4 Draw Left hand landmark on the frame
        mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Display the annotated frame
        cv2.imshow('Holistic Model Detection', image_bgr)
            
        # Check for 'q' key press to exit the loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
