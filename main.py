import cv2
from playsound import playsound
import threading

# Function to play the alarm sound in a separate thread
def play_alarm():
    playsound('alarmsound.mp3')

# Load the body classifier
body_classifier = cv2.CascadeClassifier("haarcascade_fullbody.xml")
if body_classifier.empty():
    print("Error: Haar Cascade XML file not loaded correctly")
    exit()

# Load the video file
cap = cv2.VideoCapture("vtest.avi")
if not cap.isOpened():
    print("Error: Couldn't open the video file")
    exit()

# Flag to track if the alarm sound is playing
alarm_playing = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect bodies
    bodies = body_classifier.detectMultiScale(gray, 1.1, 3)
    
    # Play alarm if bodies are detected
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
        cv2.putText(frame, "Intrusion Detected", (210, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        # Start playing the alarm sound in a separate thread if not already playing
        if not alarm_playing:
            alarm_playing = True
            threading.Thread(target=play_alarm, daemon=True).start()
    
    # Show the frame with detections
    cv2.imshow("Intrusion Detection System", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
