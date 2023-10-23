import cv2
import numpy as np

# Путь к файлу с предварительно обученной моделью YOLOv5
model_path = '/lemon/Auto_Pilot/Traffic_light/yolov5/runs/train/exp/weights/best.pt'

# Загрузка модели YOLOv5
net = cv2.dnn.readNet(model_path)


# Функция для обработки изображения и распознавания светофоров
def detect_traffic_lights(frame):
    # Получение высоты и ширины кадра
    height, width, _ = frame.shape

    # Подготовка изображения для передачи в модель
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Загрузка изображения в модель
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Список для хранения результатов распознавания
    class_ids = []
    confidences = []
    boxes = []

    # Обработка результатов распознавания
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 9:  # 9 - индекс класса "светофор" в модели
                # Координаты рамки для светофора
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Координаты углов рамки
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Нахождение индексов светофоров с высокой уверенностью
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            color = (0, 255, 0)  # Зеленый для рамки светофора
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    return frame


# Запуск камеры
cap = cv2.VideoCapture(0)  # 0 - индекс встроенной камеры

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Распознавание светофоров
    frame = detect_traffic_lights(frame)

    # Отображение кадра
    cv2.imshow('Traffic Lights Detection', frame)

    if cv2.waitKey(1) == 27:  # Нажмите клавишу Esc для выхода
        break

cap.release()
cv2.destroyAllWindows()
