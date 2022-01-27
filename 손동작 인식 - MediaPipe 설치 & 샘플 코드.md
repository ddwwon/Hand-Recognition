# 손동작 인식 - MediaPipe

### 공식문서

[https://google.github.io/mediapipe/solutions/hands.html](https://google.github.io/mediapipe/solutions/hands.html)

### 참고 사이트

- 1: [https://blog.naver.com/PostView.naver?blogId=chandong83&logNo=222472645564&categoryNo=81&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView](https://blog.naver.com/PostView.naver?blogId=chandong83&logNo=222472645564&categoryNo=81&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView)
- 2: [https://makernambo.com/154](https://makernambo.com/154)
- 3: [https://techtutorialsx.com/2021/04/20/python-real-time-hand-tracking/](https://techtutorialsx.com/2021/04/20/python-real-time-hand-tracking/)
    - 영어지만 상세한 설명

### 미디어 파이프 설치

1. cmd 창에 
    
    > pip install mediapipe
    > 
2. 예제 코드: 웹캠 이용하는 부분만
    
    [https://google.github.io/mediapipe/solutions/hands#python-solution-api](https://google.github.io/mediapipe/solutions/hands#python-solution-api)
    
    ```python
    import cv2
    import mediapipe as mp
    # 핸드 이미지 위에 랜드 마크 그리기 위함
    mp_drawing = mp.solutions.drawing_utils
    # 핸드 처리 
    mp_hands = mp.solutions.hands
    # 핸드 랜드마크 표시 스타일용
    drawing_styles = mp.solutions.drawing_styles
    
    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    
    '''
    손가락 솔루션 생성
    min_detection_confidence
    - 손가락 감지 최소 신뢰성 0.0 - 1.0
    - 1.0에 가까울수록 확실한 확률이 높은 손가락만 감지
    min_tracking_confidence
    - 손가락 추적 최소 신뢰성 0.0 - 1.0
    - 1.0에 가까울수록 추적율이 높은 것만 감지
    '''
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
      # 카메라 열려있을때까지 루프
      while cap.isOpened():
        # 카메라에서 사진 한장 얻기
        success, image = cap.read()
        # 사진을 얻어오지 못햇다면 
        if not success:
          # 에러 출력 후 루프 시작으로 되돌아감
          # 카메라 로딩중일 수 있기때문에 break대신 continue로 함
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue
    
        # 사진을 좌/우 반전 시킨후 BGR에서 RGB로 변환
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # 성능 향상을 위해 이미지를 쓰기 불가시켜 참조로 전달
        image.flags.writeable = False
        # 손가락 마디 검출!!
        results = hands.process(image)
    
        # 다시 이미지를 쓰기 가능으로 변경
        image.flags.writeable = True
        # 이미지를 다시 RGB에서 BGR로 변경
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # hands.process에서 랜드마크를 찾았다면
        if results.multi_hand_landmarks:
          # 랜드마크 숫자 만큼 루프
          for hand_landmarks in results.multi_hand_landmarks:
            # 이미지에 랜드마크 위치마다 표시
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmark_style(),
                drawing_styles.get_default_hand_connection_style())
        # 최종 변경된 이미지를 화면에 출력
        cv2.imshow('MediaPipe Hands', image)
    
        # 5ms 대기하면서 키보드 입력 대기
        # 27 > ESC 키     
        if cv2.waitKey(5) & 0xFF == 27:
          break
    cap.release()
    ```
    
3. cmd 창에
    
    > python ‘파일명’.py
    > 
4. 결과
    
    ![Untitled](%E1%84%89%E1%85%A9%E1%86%AB%E1%84%83%E1%85%A9%E1%86%BC%E1%84%8C%E1%85%A1%E1%86%A8%20%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%89%E1%85%B5%E1%86%A8%20-%20MediaPipe%207041cae26b804ada99c728415f707e90/Untitled.png)