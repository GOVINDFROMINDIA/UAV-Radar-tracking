import cv2
import numpy as np

cap = cv2.VideoCapture('jet1.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break

    jet_cascade = cv2.CascadeClassifier('jet.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    jets = jet_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in jets:
        target_size = min(w, h) // 2
        target_x = x + w // 2
        target_y = y + h // 2
        cv2.drawMarker(frame, (target_x, target_y), (0, 0, 255),
                       cv2.MARKER_CROSS, markerSize=target_size,
                       thickness=2)
        speed_box_width = 50
        speed_box_height = 20
        speed_box_x = x
        speed_box_y = y - speed_box_height - 5
        cv2.rectangle(frame, (speed_box_x, speed_box_y),
                      (speed_box_x + speed_box_width, speed_box_y + speed_box_height),
                      (0, 0, 0), -1)

        speed = 100
        speed_text = '{:.0f} km/h'.format(speed)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size, _ = cv2.getTextSize(speed_text, font, font_scale, font_thickness)
        text_x = speed_box_x + (speed_box_width - text_size[0]) // 2
        text_y = speed_box_y + (speed_box_height + text_size[1]) // 2
        cv2.putText(frame, speed_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    cv2.imshow('Jet Tracker', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
