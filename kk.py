import cv2
import mediapipe as mp
import pygame
import numpy as np
import sys
import os
import time

# Initialize Pygame mixer
try:
    pygame.mixer.init()
except pygame.error as e:
    print(f"Failed to initialize Pygame mixer: {e}")
    sys.exit(1)

# Load audio files
note_files = {
    'C': 'C.wav',
    'D': 'D.wav',
    'E': 'E.wav',
    'F': 'F.wav',
    'G': 'G.wav',
    'A': 'A.wav',
    'B': 'B.wav',
    'C1': 'C1.wav'
}

notes = {}
for note, file in note_files.items():
    if os.path.exists(file):
        try:
            notes[note] = pygame.mixer.Sound(file)
        except pygame.error as e:
            print(f"Error loading {file}: {e}")
            notes[note] = None
    else:
        print(f"Audio file {file} not found.")
        notes[note] = None

# Initialize finger states
finger_states = {note: False for note in note_files.keys()}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

def play_note_based_on_fingers(hand_landmarks, hand_index):
    global finger_states
    # Define finger tip and pip landmarks
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    notes_order = ['C', 'D', 'E', 'F'] if hand_index == 0 else ['G', 'A', 'B', 'C1']

    for tip, pip, note in zip(finger_tips, finger_pips, notes_order):
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
            if not finger_states[note]:
                if notes[note]:
                    notes[note].play()
                finger_states[note] = True
        else:
            finger_states[note] = False

def is_fist(hand_landmarks):
    folded_fingers = 0
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
            folded_fingers += 1
    return folded_fingers == 4

def display_countdown(img, countdown_start=3):
    font = cv2.FONT_HERSHEY_SIMPLEX
    y0, dy = img.shape[0] // 2, 50  # Starting position for the countdown
    for i in range(countdown_start, 0, -1):
        img_copy = img.copy()
        text = f"{i}"
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        x = (img.shape[1] - textsize[0]) // 2  # Center the text horizontally
        y = y0 + (countdown_start - i) * dy
        cv2.putText(img_copy, text, (x, y), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Hand Gesture Music Player', img_copy)
        cv2.waitKey(1000)  # Wait for 1 second

    # Display 'Bye-bye'
    img_copy = img.copy()
    text = "Bye-bye"
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    x = (img.shape[1] - textsize[0]) // 2  # Center the text horizontally
    y = y0 + (countdown_start - 1) * dy
    cv2.putText(img_copy, text, (x, y), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Hand Gesture Music Player', img_copy)
    cv2.waitKey(1000)  # Wait for 1 second


# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    sys.exit()

# Create a named window
cv2.namedWindow('Hand Gesture Music Player', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Hand Gesture Music Player', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    img = cv2.flip(img, 1)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    results = hands.process(image_rgb)

    # Draw hand annotations on the image
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_label = handedness.classification[0].label
            hand_index = 0 if hand_label == 'Left' else 1
            play_note_based_on_fingers(hand_landmarks, hand_index)
            if hand_label == 'Right' and is_fist(hand_landmarks):
                print("Right hand fist detected. Closing application in 3 seconds...")
                display_countdown(img)
                cap.release()
                cv2.destroyAllWindows()
                sys.exit()

    cv2.imshow('Hand Gesture Music Player', img)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
