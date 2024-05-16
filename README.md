Đoạn mã sử dụng thư viện OpenCV và MediaPipe để nhận diện bàn tay từ camera, tính toán khoảng cách giữa ngón tay cái và ngón tay trỏ để điều chỉnh độ sáng của màn hình.

### Tải các thư viện cần thiết từ file requirements.txt
```python
pip install -r requirements.txt
```

### Khởi tạo và nhập các thư viện
```python
import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np
```
- `cv2`: Thư viện OpenCV để xử lý hình ảnh.
- `mediapipe`: Thư viện MediaPipe để xử lý các điểm đặc trưng của bàn tay.
- `hypot`: Hàm từ thư viện `math` để tính toán khoảng cách Euclidean.
- `screen_brightness_control`: Thư viện để điều khiển độ sáng màn hình.
- `numpy`: Thư viện numpy để xử lý, làm việc với mảng và các tính toán số học.

### Khởi tạo mô hình
```python
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2
)
Draw = mp.solutions.drawing_utils
```
- `mpHands`: Module của MediaPipe để xử lý bàn tay.
- `hands`: Tạo một đối tượng `Hands` từ `mpHands` để nhận diện bàn tay trong video.
- `Draw`: Công cụ vẽ từ MediaPipe để vẽ các điểm đặc trưng của bàn tay lên khung hình.

### Mở camera và bắt đầu quá trình xử lý video
```python
cap = cv2.VideoCapture(0)
```
- `cap`: Đối tượng để bắt đầu thu hình từ webcam. `0` là chỉ số của webcam mặc định.

### Vòng lặp chính để xử lý từng khung hình
```python
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Process = hands.process(frameRGB)
    landmarkList = []
```
- `_, frame = cap.read()`: Đọc một khung hình từ webcam.
- `frame = cv2.flip(frame, 1)`: Lật ngược khung hình theo trục dọc để có hiệu ứng như gương.
- `frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`: Chuyển đổi khung hình từ BGR sang RGB vì MediaPipe yêu cầu định dạng này.
- `Process = hands.process(frameRGB)`: Xử lý khung hình để nhận diện bàn tay và các điểm đặc trưng của nó.
- `landmarkList = []`: Khởi tạo một danh sách để lưu trữ các điểm đặc trưng của bàn tay.

### Kiểm tra và xử lý các điểm đặc trưng của bàn tay
```python
if Process.multi_hand_landmarks:
    for handlm in Process.multi_hand_landmarks:
        for _id, landmarks in enumerate(handlm.landmark):
            height, width, color_channels = frame.shape
            x, y = int(landmarks.x*width), int(landmarks.y*height)
            landmarkList.append([_id, x, y])
        Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)
```
- `if Process.multi_hand_landmarks`: Kiểm tra xem có bàn tay nào được nhận diện hay không.
- `for handlm in Process.multi_hand_landmarks`: Lặp qua từng bàn tay được nhận diện.
- `for _id, landmarks in enumerate(handlm.landmark)`: Lặp qua từng điểm đặc trưng của bàn tay.
- `height, width, color_channels = frame.shape`: Lấy kích thước khung hình.
- `x, y = int(landmarks.x*width), int(landmarks.y*height)`: Chuyển đổi tọa độ từ tỷ lệ (normalized coordinates) sang tọa độ pixel.
- `landmarkList.append([_id, x, y])`: Lưu các tọa độ điểm đặc trưng vào danh sách.
- `Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)`: Vẽ các điểm đặc trưng lên khung hình.

### Tính toán khoảng cách và điều chỉnh độ sáng màn hình
```python
if landmarkList != []:
    x_1, y_1 = landmarkList[4][1], landmarkList[4][2]
    x_2, y_2 = landmarkList[20][1], landmarkList[20][2]
    cv2.circle(frame, (x_1, y_1), 7, (0, 255, 0), cv2.FILLED)
    cv2.circle(frame, (x_2, y_2), 7, (0, 255, 0), cv2.FILLED)
    cv2.line(frame, (x_1, y_1), (x_2, y_2), (0, 255, 0), 3)
    L = hypot(x_2-x_1, y_2-y_1)
    b_level = np.interp(L, [15, 220], [0, 100])
    cv2.putText(frame,f"{int(b_level)}%",(10,40),cv2.FONT_ITALIC,1,(0, 255, 98),3)
    sbc.set_brightness(int(b_level))
```
- `if landmarkList != []`: Kiểm tra xem danh sách các điểm đặc trưng có trống hay không.
- `x_1, y_1 = landmarkList[4][1], landmarkList[4][2]`: Lấy tọa độ của đầu ngón tay cái.
- `x_2, y_2 = landmarkList[20][1], landmarkList[20][2]`: Lấy tọa độ của đầu ngón tay trỏ.
- `cv2.circle(frame, (x_1, y_1), 7, (0, 255, 0), cv2.FILLED)`: Vẽ vòng tròn tại đầu ngón tay cái.
- `cv2.circle(frame, (x_2, y_2), 7, (0, 255, 0), cv2.FILLED)`: Vẽ vòng tròn tại đầu ngón tay trỏ.
- `cv2.line(frame, (x_1, y_1), (x_2, y_2), (0, 255, 0), 3)`: Vẽ đường nối giữa đầu ngón tay cái và ngón tay trỏ.
- `L = hypot(x_2-x_1, y_2-y_1)`: Tính khoảng cách giữa đầu ngón tay cái và ngón tay trỏ.
- `b_level = np.interp(L, [15, 220], [0, 100])`: Sử dụng nội suy tuyến tính để chuyển đổi khoảng cách sang mức độ sáng.
- `cv2.putText(frame,f"{int(b_level)}%",(10,40),cv2.FONT_ITALIC,1,(0, 255, 98),3)`: Hiển thị mức độ sáng trên khung hình.
- `sbc.set_brightness(int(b_level))`: Điều chỉnh độ sáng của màn hình.

### Hiển thị khung hình và thoát khi nhấn 'q'
```python
cv2.imshow('Image', frame)
if cv2.waitKey(1) & 0xff == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
```
- `cv2.imshow('Image', frame)`: Hiển thị khung hình đã qua xử lý.
- `if cv2.waitKey(1) & 0xff == ord('q')`: Kiểm tra xem người dùng có nhấn phím 'q' để thoát hay không.
- `cap.release()`: Giải phóng camera.
- `cv2.destroyAllWindows()`: Đóng tất cả các cửa sổ OpenCV.

**Copyright by Le Khac Thienn =))**
