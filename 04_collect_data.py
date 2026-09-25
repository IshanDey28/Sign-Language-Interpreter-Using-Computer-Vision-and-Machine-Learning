import cv2
import mediapipe as mp
import csv
import os

# =========================
# Ask for sign label
# =========================

LABEL = input("Enter the sign/letter to collect: ").strip().upper()

if not LABEL:
    print("No label entered. Exiting.")
    exit()

print(f"\nCollecting samples for: {LABEL}")
print("SPACE = save sample")
print("Q = quit")


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
# Camera
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera.")
    exit()


# =========================
# Dataset setup
# =========================

os.makedirs("data", exist_ok=True)

filename = "data/sign_data.csv"

file_exists = os.path.exists(filename)

sample_count = 0


# =========================
# Count existing samples
# =========================

if file_exists:

    with open(filename, "r", newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            if row["label"].strip().upper() == LABEL:
                sample_count += 1


# =========================
# Open CSV
# =========================

with open(filename, "a", newline="") as f:

    writer = csv.writer(f)

    # Create header if file doesn't exist
    if not file_exists:

        header = ["label"]

        for i in range(21):

            header += [
                f"x{i}",
                f"y{i}",
                f"z{i}"
            ]

        writer.writerow(header)


    # =========================
    # Main loop
    # =========================

    while True:

        success, frame = cap.read()

        if not success:

            print("Could not access camera.")
            break


        # Mirror camera
        frame = cv2.flip(frame, 1)


        # Convert to RGB
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # MediaPipe detection
        result = hands.process(rgb)


        # =========================
        # Hand detected
        # =========================

        if result.multi_hand_landmarks:

            hand = result.multi_hand_landmarks[0]


            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )


            # Display label
            cv2.putText(
                frame,
                f"Collecting: {LABEL}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


            # Display sample count
            cv2.putText(
                frame,
                f"Samples: {sample_count}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )


        else:

            cv2.putText(
                frame,
                "No hand detected",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )


        # =========================
        # Show camera
        # =========================

        cv2.imshow(
            "Sign Language Dataset Collector",
            frame
        )


        # =========================
        # Keyboard
        # =========================

        key = cv2.waitKey(1) & 0xFF


        # =========================
        # Save sample
        # =========================

        if key == ord(" "):

            if result.multi_hand_landmarks:

                hand = result.multi_hand_landmarks[0]

                row = [LABEL]


                # Save 21 landmarks
                for landmark in hand.landmark:

                    row.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])


                writer.writerow(row)

                # Force data to disk immediately
                f.flush()

                sample_count += 1

                print(
                    f"Saved {LABEL} sample "
                    f"#{sample_count}"
                )


            else:

                print(
                    "No hand detected. "
                    "Sample not saved."
                )


        # =========================
        # Quit
        # =========================

        elif key == ord("q"):

            break


# =========================
# Cleanup
# =========================

cap.release()
cv2.destroyAllWindows()

print(
    f"\nFinished! Total {LABEL} samples: "
    f"{sample_count}"
)