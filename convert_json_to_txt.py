# Implementation of json file YOLOV5 training txt file
# Note: The height of the image needs to be customized


import json
import os


def convert(img_size, box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]

    # Convert and return to
    center_x = (x1 + x2) * 0.5 / img_size[0]
    center_y = (y1 + y2) * 0.5 / img_size[1]
    w = abs((x2 - x1)) * 1.0 / img_size[0]
    h = abs((y2 - y1)) * 1.0 / img_size[1]

    return (center_x, center_y, w, h)


def decode_json(save_path, jsonfloder_path, json_name, classes):
    txt_name = save_path + json_name[0:-5] + '.txt'
    # txt Save location

    json_path = os.path.join(json_folder_path, json_name)
    data = json.load(open(json_path, 'r'))

    img_w = 1280
    img_h = 720
    with open(txt_name, 'w') as txt_file:  # te files
        for i in data['labels']:
            if i['box2d']:  # Only applicable to the rectangular box labeling
                x1 = float(i['box2d']['x1'])
                y1 = float(i['box2d']['y1'])
                x2 = float(i['box2d']['x2'])
                y2 = float(i['box2d']['y2'])
                if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
                    continue
                else:
                    bb = (x1, y1, x2, y2)
                    bbox = convert((img_w, img_h), bb)

                cls = i['category']  # B of current label category

                # Converted to the label read by the training mode
                cls_id = classes.index(cls)  # Located in the definition category index position

                #
                txt_file.write(str(cls_id) + ' ' + " ".join([str(a) for a in bbox]) + "\n")  # 0 CX, Cy, W, H


if __name__ == "__main__":

    #   category
    classes_train = ['pedestrian']  # Modify 1, category

    json_folder_path = 'path_1/'  # Modify 2, JSON folder path,
    save_path = 'path_2/'  # Modify 3, save position

    json_names = os.listdir(json_folder_path)  # file name

    # All JSON files
    for json_name in json_names:  # output all files
        if json_name[-5:] == '.json':  # just work for json files
            decode_json(save_path, json_folder_path, json_name, classes_train)


