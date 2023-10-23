import torch
import cv2
import numpy as np

# Путь к вашей предварительно обученной модели YOLOv5
model_path = '/lemon/Auto_Pilot/Traffic_light/yolov5/runs/train/exp/weights/best.pt'

# Загрузка модели
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Захват видеопотока с камеры
cap = cv2.VideoCapture(0)  # Индекс 0 соответствует встроенной камере

while True:
    ret, frame = cap.read()  # Захват кадра с камеры
    if not ret:
        break

    # Передача кадра в модель для распознавания
    results = model(frame)

    # Вывод результатов на экран
    print(results)

    # Отображение кадра с выделенными светофорами
    frame_with_traffic_lights = results.render()[0]

    # Отображение кадра
    cv2.imshow('Camera Feed', frame_with_traffic_lights)

    if cv2.waitKey(1) == 27:  # Для выхода нажмите клавишу Esc
        break

cap.release()
cv2.destroyAllWindows()
