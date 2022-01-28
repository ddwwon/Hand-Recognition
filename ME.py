import cv2
import mediapipe as mp
import math

# 핸드 이미지 위에 랜드마크 그리기
mp_drawing = mp.solutions.drawing_utils
# 핸드 처리
mp_hands = mp.solutions.hands
# 핸드 랜드마크 표시 스타일용
drawing_styles = mp.solutions.drawing_styles

# 웹캠 열기
cap = cv2.VideoCapture(0)

def dist(x1, y1, x2, y2):
    # 점과 점 사이 거리 공식
    return math.sqrt(math.pow(x1 - x2, 2)) + math.sqrt(math.pow(y1 - y2, 2))

compareIndex = [[18, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
open = [False, False, False, False, False]
gesture = [
    [True, True, True, True, True, "Hi!"],
    [False, True, True, False, False, "V"],
    [True, True, False, False, True, "Nikonikoni~"],
    [True, False, False, False, False, "1"],
    [False, True, False, False, False, "2"],
    [False, False, True, False, False, "3"],
    [False, False, False, True, False, "4"],
    [False, False, False, False, True, "5"],
    [False, True, False, False, True, "Yaaooo"],
    [False, False, False, False, False, "rock!"]
]

with mp_hands.Hands(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as hands:
    while cap.isOpened:
        # 카메라에서 사진 한장 얻기
        success, image = cap.read()
        h, w, c = image.shape
        if not success:
            print("error")
            continue

        image = cv2.cvtColor(cv2.flip(image, 2), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                for i in range(0, 5):
                    open[i] = dist(hand_landmark.landmark[0].x, hand_landmark.landmark[0].y, hand_landmark.landmark[compareIndex[i][0]].x,
                                   hand_landmark.landmark[compareIndex[i][0]].y) < dist(hand_landmark.landmark[0].x,
                                                                                  hand_landmark.landmark[0].y,
                                                                                  hand_landmark.landmark[
                                                                                      compareIndex[i][1]].x,
                                                                                  hand_landmark.landmark[
                                                                                      compareIndex[i][1]].y)
                    # print(open)
                    test_x = (hand_landmark.landmark[0].x * w)
                    test_y = (hand_landmark.landmark[0].y * h)
                    for i in range(0, len(gesture)):
                        flag = True
                        for j in range(0, 5):
                            if (gesture[i][j] != open[j]):
                                flag = False
                        if (flag == True):
                            cv2.putText(image, gesture[i][5], (round(test_x) - 50, round(test_y) - 250),
                                        cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
                    mp_drawing.draw_landmarks(image, hand_landmark, mp_hands .HAND_CONNECTIONS)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()