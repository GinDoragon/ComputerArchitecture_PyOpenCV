import cv2
import numpy as np
import os
import time

# ������� ����� ��� ���������� �����
output_dir = "output_videos"
os.makedirs(output_dir, exist_ok=True)

# ������������� ������������
cap = cv2.VideoCapture(0)

# �������� ���� � ���������� ����������
cv2.namedWindow('Controls')
cv2.resizeWindow('Controls', 400, 350)

# �������� ���������
cv2.createTrackbar('Motion Detect', 'Controls', 0, 1, lambda x: None)
cv2.createTrackbar('Min Area', 'Controls', 100, 5000, lambda x: None)
cv2.createTrackbar('Bg Threshold', 'Controls', 16, 100, lambda x: None)
cv2.createTrackbar('Blur Level', 'Controls', 0, 10, lambda x: None)
cv2.createTrackbar('Saturation (%)', 'Controls', 100, 200, lambda x: None)
cv2.createTrackbar('Contrast (%)', 'Controls', 100, 200, lambda x: None)
cv2.createTrackbar('Brightness', 'Controls', 100, 200, lambda x: None)
cv2.createTrackbar('Record', 'Controls', 0, 1, lambda x: None)

# ������������� ��������� ����
subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# ��������� �����
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
recording = False
out = None
prev_record_state = 0

# ���� ��� ��������������� ��������
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    try:
        # ��������� �������� � ���������
        motion_on = cv2.getTrackbarPos('Motion Detect', 'Controls')
        min_area = cv2.getTrackbarPos('Min Area', 'Controls')
        bg_threshold = cv2.getTrackbarPos('Bg Threshold', 'Controls')
        blur_level = cv2.getTrackbarPos('Blur Level', 'Controls')
        saturation = cv2.getTrackbarPos('Saturation (%)', 'Controls') / 100.0
        contrast = cv2.getTrackbarPos('Contrast (%)', 'Controls') / 100.0
        brightness = (cv2.getTrackbarPos('Brightness', 'Controls') - 100) * 2
        record_state = cv2.getTrackbarPos('Record', 'Controls')
    except Exception as e:
        print ("Controls not detected", e)
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
    # ���������� ���������� ���������
    subtractor.setVarThreshold(bg_threshold)

    processed_frame = frame.copy()
    
    # ����������� ��������
    if motion_on:
        fg_mask = subtractor.apply(frame)
        
        # ��������������� �������� ��� �������� ����
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # ����� ��������
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # ��������� ��������������� ������ ��������
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # ���������� ��������
    # ��������
    if blur_level > 0:
        kernel_size = blur_level * 2 + 1
        processed_frame = cv2.GaussianBlur(processed_frame, (kernel_size, kernel_size), 0)
    
    # ������������
    hsv = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype("float32")
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    processed_frame = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    
    # �������� � �������
    processed_frame = cv2.convertScaleAbs(processed_frame, alpha=contrast, beta=brightness)

    # ���������� �������
    if record_state != prev_record_state:
        if record_state == 1:
            codec = cv2.VideoWriter_fourcc(*'XVID')
            filename = os.path.join(output_dir, f"recording_{time.strftime('%Y%m%d_%H%M%S')}.avi")
            out = cv2.VideoWriter(filename, codec, 20.0, (frame_width, frame_height))
            recording = True
            print(f"Start recording: {filename}")
        else:
            if out is not None:
                out.release()
                out = None
                recording = False
                print("Recording stopped")
        prev_record_state = record_state

    # ������ �����
    if recording:
        out.write(processed_frame)
        cv2.putText(processed_frame, "RECORDING", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ����������� ����������
    info = [
        f"Motion Detect: {'ON' if motion_on else 'OFF'}",
        f"Blur: {blur_level}",
        f"Saturation: {saturation:.1f}",
        f"Contrast: {contrast:.1f}",
        f"Brightness: {brightness}",
        f"Min Area: {min_area}",
        f"Bg Threshold: {bg_threshold}"
        #f"Detector: {'somebody detected' if contours.count > 0 else 'nobody detected'}"
    ]
    y = 30
    for text in info:
        cv2.putText(processed_frame, text, (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        y += 25

    # ����������� �����
    cv2.imshow('Camera Feed', processed_frame)

    # ����� �� ������� ESC
    if cv2.waitKey(1) == 27:
        break

# ������������ ��������
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()