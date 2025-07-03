import numpy as np
from matplotlib import pyplot as plt
import cv2
import tensorflow.compat.v1 as tf
import math
import os
import json
import random


class VIA_COCO:
    def __init__(self,dir,):
        '''
        dir: the path of tfrecord
        '''
        self.tfrcd_file = dir

    @staticmethod
    def delete_inf_list(data_list):
        """
        Extract inf and nan from data list

        Args:
            data_list:  list of data
        Return:
            data_list:  list of data without inf and nan
        """
        del_list = []

        for index, item in enumerate(data_list):
            if (math.isinf(item)) or (math.isnan(item)):
                del_list.append(index)

        if len(del_list) != 0:
            for index in sorted(del_list, reverse=True):
                del data_list[index]

        return data_list

    def extract(self,output_path, isSplit):



# classes
via_classes = ["bike", "bus", "car", "motor", "person", "truck", "rider",
               "traffic sign", "delineator", "traffic light red", "traffic light green", "traffic light yellow",
               "thermal_bus", "thermal_car", "thermal_truck", "thermal_person",
               "sl60", "sl70", "sl80", "sl90", "sl100", "sl110",
               "stop", "sl25", "trolley"]

# coco format
coco = {
    'info': 'Hess_20241206',
    'videos': [],
    'images': [],
    'annotations': [],
    "categories": [{'supercategorie': '', 'id': 0, 'name': 'bike'},
                   {'supercategorie': '', 'id': 1, 'name': 'bus'},
                   {'supercategorie': '', 'id': 2, 'name': 'car'},
                   {'supercategorie': '', 'id': 3, 'name': 'motor'},
                   {'supercategorie': '', 'id': 4, 'name': 'person'},
                   {'supercategorie': '', 'id': 5, 'name': 'truck'},
                   {'supercategorie': '', 'id': 6, 'name': 'traffic light'},
                   {'supercategorie': '', 'id': 7, 'name': 'traffic sign'},
                   {'supercategorie': '', 'id': 8, 'name': 'delineator'},
                   {'supercategorie': '', 'id': 9, 'name': 'traffic light red'},
                   {'supercategorie': '', 'id': 10, 'name': 'traffic light green'},
                   {'supercategorie': '', 'id': 11, 'name': 'traffic light yellow'},
                   {'supercategorie': '', 'id': 12, 'name': 'thermal_bus'},
                   {'supercategorie': '', 'id': 13, 'name': 'thermal_car'},
                   {'supercategorie': '', 'id': 14, 'name': 'thermal_truck'},
                   {'supercategorie': '', 'id': 15, 'name': 'thermal_person'},
                   {'supercategorie': '', 'id': 16, 'name': 'sl60'},
                   {'supercategorie': '', 'id': 17, 'name': 'sl70'},
                   {'supercategorie': '', 'id': 18, 'name': 'sl80'},
                   {'supercategorie': '', 'id': 19, 'name': 'sl90'},
                   {'supercategorie': '', 'id': 20, 'name': 'sl100'},
                   {'supercategorie': '', 'id': 21, 'name': 'sl110'},
                   {'supercategorie': '', 'id': 22, 'name': 'stop'},
                   {'supercategorie': '', 'id': 23, 'name': 'sl25'},
                   {'supercategorie': '', 'id': 24, 'name': 'trolley'}]
}

# data path
data_dir = '/dataset/VIA/'
input_file_path = data_dir + 'tf_record/20241121-aws.tfrecord'
output_path = data_dir + 'coco_format/val/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

tfrecords_filename = input_file_path

# read file and init
record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
vid_idx = 0
frame_idx = 0
img_idx = 0
obj_idx = 0

for string_record in record_iterator:
    # build Example
    example = tf.train.Example()

    # extract information from example
    example.ParseFromString(string_record)

    # extract height Feature
    height = int(example.features.feature['image/height'].int64_list.value[0])

    # extract width Feature
    width = int(example.features.feature['image/width'].int64_list.value[0])
    print('picure height  and width', height, width)

    # extract file name
    filename = example.features.feature['image/filename'].bytes_list.value[0].decode('utf-8')
    # print(filename)
    clip_name = filename[-10:]
    # print(clip_name)
    # vid_name = os.path.dirname(filename)
    # print(vid_name)

    # soiling = int(example.features.feature['image/attribute/lens/soiling'].int64_list.value[0])
    # print("Length of list:", img_idx)
    # if soiling == 1:
    #     continue
    # extract image_encoded Feature
    image_decode = tf.image.decode_jpeg(example.features.feature['image/encoded'].bytes_list
                                        .value[0], channels=3)

    # extract label Feature
    label = (example.features.feature['image/object/class/label'].int64_list.value)
    # print('label is ',label)
    soiling = (example.features.feature['image/attribute/lens/soiling'].int64_list.value)
    # print( "test_soiling: "+str(soiling))
    # extract text Feature
    text = (example.features.feature['image/object/class/text'].bytes_list.value)
    # print('label is ', label)

    # extract bbox Feature
    xmin_array = np.array(list(example.features.feature['image/object/bbox/xmin'].float_list.value))
    xmax_array = np.array(list(example.features.feature['image/object/bbox/xmax'].float_list.value))
    ymin_array = np.array(list(example.features.feature['image/object/bbox/ymin'].float_list.value))
    ymax_array = np.array(list(example.features.feature['image/object/bbox/ymax'].float_list.value))

    if clip_name == '000001.png':
        # coco['videos'].append({
        #     "name": f'video{vid_idx:06d}',
        #     "id": vid_idx,
        #     "width":1280,
        #     "height":720,
        # }, )
        all_aspect_ratio = []
        frame_idx = 0
        vid_idx += 1

    image_rgb = cv2.cvtColor(image_decode.numpy(), cv2.COLOR_BGR2RGB)

    isCrowded = 1 if len(text) > 1 else 0

    for i in range(len(text)):
        obj_idx += 1
        # print('element of box',obj_idx)
        xmin = xmin_array[i] * width
        xmax = xmax_array[i] * width
        ymin = ymin_array[i] * height
        ymax = ymax_array[i] * height
        # pt1 = int(xmin), int(ymin)
        # pt2 = int(xmax), int(ymax)
        # cv2.rectangle(image_rgb, pt1, pt2, (0, 255, 0), 2)
        # pt1 = int(xmin), int(ymax)
        # pt2 = int(xmin), int(ymax + 20)
        # cv2.rectangle(image_rgb, pt1, pt2, (125, 125, 125), -1)
        # pt2 = int(xmin), int(ymax + 15)
        # print(label[i])
        # lable the tag to picture
        # cv2.putText(image_rgb, via_classes[label[i]], pt2,
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        # print(via_classes[label[i]])

        # cv2.putText(image_rgb, str(text[i]), pt2,
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        coco['annotations'].append(
        {"segmentations": [],
         "area": (ymax - ymin+ 1) * (xmax - xmin +1),
         "iscrowd": isCrowded,
         "image_id":img_idx ,
         "bbox": [xmin, ymin, xmax - xmin + 1, ymax - ymin + 1],
         "category_id": label[i],
         "id": obj_idx})

    name = (example.features.feature['image/filename'].bytes_list)
    # img_name = "test_image_"+str(index) +'_'+ str(name)[-40:]
    # print(img_name)
    # lens = example.features.feature['image/attribute/lens/soiling'].int64_list.value[0]
    # if (lens > 1):
    #     # continue
    #     cv2.putText(image_rgb, 'occluded', (20, 50),
    #                 cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
    # else:
    #     if (lens>0):
    #         cv2.putText(image_rgb, 'soiling', (20, 50),
    #                     cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)
    #         # cv2.putText(image_rgb, str(lens), (20, 80),
    #         #             cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    #     else:
    #         # continue
    #         cv2.putText(image_rgb, 'clear', (20, 50),
    #                     cv2.FONT_HERSHEY_COMPLEX, 1 , (0, 0, 255), 1)
    coco['images'].append({
        'file_name':f'Data/video{vid_idx:06d}/' + f'image{img_idx:06d}.JPEG',
        'height':720,
        'width':1280,
        'id':img_idx,
        'frame_id':frame_idx,
        'video_id':vid_idx-1, # when initialize, it adds 1
    })

    if not os.path.exists(output_path + f'Data/video{vid_idx:06d}/'):
        os.makedirs(output_path + f'Data/video{vid_idx:06d}/')
    # print(output_path+f'Data/video{vid_idx:06d}/'+f'image{img_idx:06d}.JPEG')
    # in the folder
    cv2.imwrite(output_path + f'Data/video{vid_idx:06d}/' + f'image{img_idx:06d}.JPEG', image_rgb)

    print(filename, '\n', f'Data/video{vid_idx:06d}/' + f'image{img_idx:06d}.JPEG', [via_classes[label[i]] for i in range(len(text))])
    img_idx += 1
    frame_idx += 1
    # all pics stacked
    # cv2.imwrite(output_path + img_name, image_rgb)

# flat_list = [item for sublist in all_aspect_ratio for item in sublist]

# for debug
# print (np.sum(np.isinf(flat_list)))

# flat_list = delete_inf_list(flat_list)
# print ("Length of list:",len(flat_list))

print(coco)


if not os.path.exists(output_path+ '/Annotations/'):
    os.makedirs(output_path+ '/Annotations/')
with open(output_path+ '/Annotations/20241121coco.json', 'w') as f:
    json.dump(coco, f)


def split_coco_annotation(json_path, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    with open(json_path, 'r') as f:
        coco = json.load(f)

    images = coco['images']
    annotations = coco['annotations']
    categories = coco['categories']

    # 固定随机种子以保证可复现
    random.seed(seed)
    random.shuffle(images)

    # 计算切分点
    total = len(images)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    def filter_annotations(images_subset):
        image_ids = set(img['id'] for img in images_subset)
        return [ann for ann in annotations if ann['image_id'] in image_ids]

    # 构建子集
    subsets = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }

    os.makedirs(output_dir, exist_ok=True)

    for subset, imgs in subsets.items():
        subset_json = {
            'images': imgs,
            'annotations': filter_annotations(imgs),
            'categories': categories
        }

        out_path = os.path.join(output_dir, f'instances_{subset}.json')
        with open(out_path, 'w') as f:
            json.dump(subset_json, f)
        print(f"Saved {subset} set with {len(imgs)} images and {len(subset_json['annotations'])} annotations to {out_path}")

# 用法示例
# split_coco_annotation(
#     json_path=output_path+ '/Annotations/20241206coco.json',    # 原始 COCO 文件
#     output_dir=output_path+ '/Annotations/',                # 输出目录
#     train_ratio=0.8,
#     val_ratio=0.1,
#     test_ratio=0.1
# )