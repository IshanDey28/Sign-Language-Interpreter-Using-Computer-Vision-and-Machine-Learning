import cv2
import mediapipe as mp
import joblib
import numpy as np

# =========================
# Load trained model
# =========================

model = joblib.load("models/sign_model.pkl")

# =========================
# MediaPipe setup
# =========================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =========================
# Start camera
# =========================

cap = cv2.VideoCapture(0)

print("Sign Language Reader started.")
print("Press Q to quit.")

while True:

    success, frame = cap.read()

    if not success:
        print("Could not access camera.")
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand
    result = hands.process(rgb)

    prediction = ""
    confidence = 0

    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        # =========================
        # Extract 21 landmarks
        # =========================

        features = []

        for landmark in hand.landmark:

            features.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        # Convert to NumPy array
        features = np.array(features).reshape(1, -1)

        # =========================
        # Predict sign
        # =========================

        prediction = model.predict(features)[0]

        # Get probability/confidence
        probabilities = model.predict_proba(features)[0]

        confidence = np.max(probabilities)

        # =========================
        # Display prediction
        # =========================

        cv2.putText(
            frame,
            f"Sign: {prediction}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence * 100:.1f}%",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "No hand detected",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # Show camera
    cv2.imshow("ASL Sign Language Reader", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()