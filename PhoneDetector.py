# Import required libraries
import cv2  # OpenCV for camera
import mediapipe as mp  # MediaPipe for face detection
import time  # For cooldown timer
import pygame  # For playing sound
import requests  # For downloading sound
import threading  # For playing media in background
import os  # For file operations
import numpy as np  # For audio generation

# Initialize pygame mixer for sound
pygame.mixer.init()

# Initialize MediaPipe Face Landmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Download face landmarker model if not exists
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading face detection model...")
    url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
    response = requests.get(url)
    with open(model_path, 'wb') as f:
        f.write(response.content)
    print("Model downloaded!")

# Configure face landmarker
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Create face landmarker
landmarker = FaceLandmarker.create_from_options(options)

# Initialize webcam
cap = cv2.VideoCapture(0)

# State variables
looking_down = False
start_time = None
triggered = False
last_trigger_time = 0
COOLDOWN_SECONDS = 10
frame_count = 0

def calculate_head_tilt(face_landmarks):
    # Get multiple face landmarks for better detection
    nose_tip = face_landmarks[1]
    chin = face_landmarks[152]
    forehead = face_landmarks[10]
    left_eye = face_landmarks[33]
    right_eye = face_landmarks[263]
    
    # Calculate head tilt using multiple points
    # When looking down, forehead y is LOWER than nose y (negative tilt)
    head_tilt = (forehead.y - nose_tip.y)
    
    # Also check eye position relative to nose
    eye_center_y = (left_eye.y + right_eye.y) / 2
    eye_to_nose = nose_tip.y - eye_center_y
    
    # Check nose to chin distance (gets smaller when looking down)
    nose_chin_dist = chin.y - nose_tip.y
    
    return head_tilt, eye_to_nose, nose_chin_dist

def play_scream():
    # Generate loud beep sound as scream
    try:
        print("Playing scream sound!")
        
        # Generate a loud beeping sound
        duration = 2000  # 2 seconds
        frequency = 800  # Hz
        
        # Create sound array
        sample_rate = 22050
        samples = int(sample_rate * duration / 1000)
        
        # Generate square wave (harsh sound)
        wave = np.array([np.sin(2 * np.pi * frequency * t / sample_rate) for t in range(samples)])
        wave = (wave * 32767).astype(np.int16)
        
        # Create stereo sound
        stereo_wave = np.column_stack((wave, wave))
        
        # Play using pygame
        sound = pygame.sndarray.make_sound(stereo_wave)
        sound.set_volume(1.0)  # Max volume
        sound.play()
        
        print("Sound playing!")
        
    except Exception as e:
        print(f"Error: {e}")
        # Fallback - system beep
        import winsound
        winsound.Beep(1000, 2000)  # 1000 Hz for 2 seconds

def show_scream_effect(img):
    # Show red flashing effect for 3 seconds
    try:
        for i in range(90):  # 3 seconds at 30fps
            # Create red overlay
            scream_img = img.copy()
            red_overlay = scream_img.copy()
            red_overlay[:] = (0, 0, 255)
            
            # Flash effect - alternate intensity
            alpha = 0.8 if i % 10 < 5 else 0.5
            scream_img = cv2.addWeighted(scream_img, 1-alpha, red_overlay, alpha, 0)
            
            # Add text
            cv2.putText(scream_img, "YOU GOT CAUGHT!", (60, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
            cv2.putText(scream_img, "STOP USING YOUR PHONE!", (20, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            
            cv2.imshow('Phone Detector', scream_img)
            cv2.waitKey(33)  # ~30fps
            
    except Exception as e:
        print(f"Error showing effect: {e}")

# Main loop
print("Phone Detection Active!")
print("Look down for 2 seconds to trigger!")

import random

while True:
    # Read frame from camera
    success, img = cap.read()
    if not success:
        continue
    
    frame_count += 1
    
    # Flip image for mirror effect
    img = cv2.flip(img, 1)
    
    # Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    # Detect face landmarks
    results = landmarker.detect_for_video(mp_image, frame_count)
    
    current_time = time.time()
    
    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            # Calculate if head is tilted down
            head_tilt, eye_to_nose, nose_chin = calculate_head_tilt(face_landmarks)
            
            # Show debug info
            cv2.putText(img, f"Tilt: {head_tilt:.3f}", (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(img, f"Eye-Nose: {eye_to_nose:.3f}", (50, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(img, f"Nose-Chin: {nose_chin:.3f}", (50, 210), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Check if looking DOWN - multiple conditions
            # When looking down: tilt is negative AND eye-nose increases AND nose-chin decreases
            is_looking_down = (head_tilt < -0.05) and (eye_to_nose > 0.1) and (nose_chin < 0.2)
            
            if is_looking_down:
                # User is looking down
                if not looking_down:
                    looking_down = True
                    start_time = current_time
                    print("Detected looking down...")
                
                # Check if looking down for 2 seconds
                time_looking_down = current_time - start_time
                
                # Draw warning on screen
                warning_text = f"Looking down! {2 - int(time_looking_down)}s"
                cv2.putText(img, warning_text, (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                # Draw warning rectangle
                cv2.rectangle(img, (40, 20), (400, 80), (0, 0, 255), 3)
                
                # Trigger scream after 2 seconds
                if time_looking_down >= 2 and not triggered:
                    # Check cooldown
                    if current_time - last_trigger_time > COOLDOWN_SECONDS:
                        print("TRIGGERED!")
                        
                        # Play scream sound in background
                        sound_thread = threading.Thread(target=play_scream)
                        sound_thread.start()
                        
                        # Show visual effect
                        show_scream_effect(img)
                        
                        triggered = True
                        last_trigger_time = current_time
            else:
                # User is looking at screen - reset
                looking_down = False
                triggered = False
                start_time = None
                
                # Show status
                cv2.putText(img, "Looking at screen - Good!", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # No face detected
        cv2.putText(img, "No face detected", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
    
    # Show cooldown timer if active
    if current_time - last_trigger_time < COOLDOWN_SECONDS:
        cooldown_remaining = int(COOLDOWN_SECONDS - (current_time - last_trigger_time))
        cv2.putText(img, f"Cooldown: {cooldown_remaining}s", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
    
    # Display the frame
    cv2.imshow('Phone Detector', img)
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
landmarker.close()
pygame.mixer.quit()
