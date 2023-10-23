import os
import xml.etree.ElementTree as ET

# Путь к папке с XML-файлами аннотаций
xml_dir = '/lemon/Auto_Pilot/raw_data_with_images_traffic_light/'

# Папка для сохранения файлов аннотаций YOLOv5
output_dir = '/lemon/Auto_Pilot/pretrained_data_traffic_light/'

# Словарь для соответствия названий классов и их числовых идентификаторов
class_dict = {}

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Функция для получения идентификатора класса
def get_class_id(class_name):
    if class_name not in class_dict:
        class_dict[class_name] = len(class_dict)
    return class_dict[class_name]

# Проходим по всем XML-файлам и создаем соответствующие файлы YOLO
for xml_file in os.listdir(xml_dir):
    if not xml_file.endswith('.xml'):
        continue

    xml_path = os.path.join(xml_dir, xml_file)
    output_path = os.path.join(output_dir, os.path.splitext(xml_file)[0] + '.txt')

    # Считываем XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    annotations = []

    for obj in root.iter('object'):
        class_name = obj.find('name').text
        class_id = get_class_id(class_name)
        x_min = int(obj.find('bndbox').find('xmin').text)
        y_min = int(obj.find('bndbox').find('ymin').text)
        x_max = int(obj.find('bndbox').find('xmax').text)
        y_max = int(obj.find('bndbox').find('ymax').text)
        x_center = (x_min + x_max) / 2.0 / width
        y_center = (y_min + y_max) / 2.0 / height
        width_box = (x_max - x_min) / width
        height_box = (y_max - y_min) / height

        annotations.append((class_id, x_center, y_center, width_box, height_box))

    with open(output_path, 'w') as f:
        for annotation in annotations:
            class_id, x_center, y_center, width_box, height_box = annotation
            line = f"{class_id} {x_center} {y_center} {width_box} {height_box}\n"
            f.write(line)

print("Конвертация завершена.")