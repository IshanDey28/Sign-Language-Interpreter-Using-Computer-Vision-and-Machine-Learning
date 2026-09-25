import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

tip_ids = [4, 8, 12, 16, 20]

def get_fingers_up(lm):
    fingers = []
    if lm[tip_ids[0]].x < lm[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    for id in range(1, 5):
        if lm[tip_ids[id]].y < lm[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def recognize_gesture(fingers):
    if fingers == [1, 1, 1, 1, 1]:
        return "HELLO"
    elif fingers == [0, 0, 0, 0, 0]:
        return "NO"
    elif fingers == [1, 0, 0, 0, 0]:
        return "YES"
    elif fingers == [0, 1, 1, 0, 0]:
        return "THANK YOU"
    elif fingers == [1, 1, 0, 0, 1]:
        return "I LOVE YOU"
    else:
        return ""

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    word = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers = get_fingers_up(hand_landmarks.landmark)
            word = recognize_gesture(fingers)

    if word:
        cv2.putText(frame, word, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Sign Language Reader", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()