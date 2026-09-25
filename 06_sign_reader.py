import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque

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
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)


# =========================
# Stability settings
# =========================

BUFFER_SIZE = 7
CONFIDENCE_THRESHOLD = 0.75

prediction_buffer = deque(maxlen=BUFFER_SIZE)

last_confirmed = ""


# =========================
# Start camera
# =========================

cap = cv2.VideoCapture(0)

print("ASL Sign Reader started.")
print("Press Q to quit.")


while True:

    success, frame = cap.read()

    if not success:
        print("Could not access camera.")
        break

    frame = cv2.flip(frame, 1)
    # Improve visibility in darker lighting
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    frame = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = hands.process(rgb)

    prediction = ""
    confidence = 0


    # =========================
    # Hand detected
    # =========================

    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )


        # =========================
        # Extract landmarks
        # =========================

        features = []

        for landmark in hand.landmark:

            features.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        features = np.array(
            features
        ).reshape(1, -1)


        # =========================
        # Model prediction
        # =========================

        probabilities = model.predict_proba(features)[0]

        confidence = np.max(probabilities)

        if confidence >= CONFIDENCE_THRESHOLD:

            prediction = model.classes_[
                np.argmax(probabilities)
            ]

        else:

            prediction = ""


        # =========================
        # Show probabilities
        # =========================

        classes = model.classes_

        for label, probability in zip(
            classes,
            probabilities
        ):

            print(
                f"{label}: {probability * 100:.1f}%",
                end=" | "
            )

        print()


        # =========================
        # Add confident prediction
        # to stability buffer
        # =========================

        if prediction:

            prediction_buffer.append(
                prediction
            )


        # =========================
        # Check stability
        # =========================

        if len(prediction_buffer) == BUFFER_SIZE:

            counts = {}

            for p in prediction_buffer:

                counts[p] = counts.get(p, 0) + 1


            most_common = max(
                counts,
                key=counts.get
            )

            occurrences = counts[
                most_common
            ]


            # Require 80% agreement

            if occurrences >= 5:

                if most_common != last_confirmed:

                    last_confirmed = most_common

                    print(
                        f"CONFIRMED SIGN: "
                        f"{last_confirmed}"
                    )


        # =========================
        # Display live prediction
        # =========================

        live_text = (
            prediction
            if prediction
            else "Uncertain"
        )

        cv2.putText(
            frame,
            f"Live: {live_text}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )


        cv2.putText(
            frame,
            f"Confidence: "
            f"{confidence * 100:.1f}%",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Confirmed: "
            f"{last_confirmed}",
            (10, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            3
        )


    else:

        # =========================
        # No hand detected
        # =========================

        prediction_buffer.clear()

        cv2.putText(
            frame,
            "No hand detected",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # =========================
    # Show camera
    # =========================

    cv2.imshow(
        "ASL Sign Language Reader",
        frame
    )


    # =========================
    # Quit
    # =========================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


cap.release()
cv2.destroyAllWindows()