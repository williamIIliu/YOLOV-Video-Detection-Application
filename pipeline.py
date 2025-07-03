from yolox.data.datasets import vid,ovis,coco
import cv2
# ovis dataset to coco
# Ovis = ovis.OVIS(
#     data_dir="/media/hazelwang/FE92D56C92D529C3/yolov/YOLOV/dataset/aws/files/" ,
#     json_file='20241119vid.json',
#     img_size=(1280,720),
# )
# ovis.convert_ovis_coco(data_dir="/media/hazelwang/FE92D56C92D529C3/yolov/YOLOV/dataset/aws/files/" + '20241119vid.json',
#                        save_dir="/media/hazelwang/FE92D56C92D529C3/yolov/YOLOV/dataset/aws/files/" + '20241119ovis.json')

'''
COCO format annotations, the annotations is saved in data_dir/annotations/json_file
the pic is listed in the data_dir rather in each vid_folder
'''
Coco = coco.COCODataset(
    data_dir="/media/hazelwang/FE92D56C92D529C3/yolov/YOLOV/dataset/aws/files/" ,
    json_file='20241119vid.json',
    img_size=(1280,720),
    name='20241119',
)
img1 = Coco.load_image(1)
cv2.imshow('window',img1)
cv2.waitKey(0)
# dataset = vid.VIDDataset()
#
# # change photos to seqs
# series = dataset.photo_to_sequence("/media/hazelwang/FE92D56C92D529C3/yolov/YOLOV/dataset/aws/files/20241119/",lframe=1280,gframe=720)
# print(series)
# annos = dataset.get_annotation("/media/hazelwang/FE92D56C92D529C3/yolov/YOLOV/dataset/aws/files/20241119/",512)
# print(annos)


# dataset = vid.Arg_VID(
#     data_dir= "/media/hazelwang/FE92D56C92D529C3/yolov/YOLOV/dataset/aws/files/20241119/",
#     img_size=(1280,720),
#     COCO_anno="/media/hazelwang/FE92D56C92D529C3/yolov/YOLOV/dataset/aws/files/20241119/",
# )

# change photos to seqs
# series = dataset.photo_to_sequence("/media/hazelwang/FE92D56C92D529C3/yolov/YOLOV/dataset/aws/files/20241119/",lframe=1280,gframe=720)
# print(series)


# coco = {
#     'info': '20241119',
#     'annotations': [],
#     "categories": [{"supercategorie": "", "id": 0, "name": "airplane"}, {"supercategorie": "", "id": 1, "name": "antelope"}, {"supercategorie": "", "id": 2, "name": "bear"}, {"supercategorie": "", "id": 3, "name": "bicycle"}, {"supercategorie": "", "id": 4, "name": "bird"}, {"supercategorie": "", "id": 5, "name": "bus"}, {"supercategorie": "", "id": 6, "name": "car"}, {"supercategorie": "", "id": 7, "name": "cattle"}, {"supercategorie": "", "id": 8, "name": "dog"}, {"supercategorie": "", "id": 9, "name": "domestic_cat"}, {"supercategorie": "", "id": 10, "name": "elephant"}, {"supercategorie": "", "id": 11, "name": "fox"}, {"supercategorie": "", "id": 12, "name": "giant_panda"}, {"supercategorie": "", "id": 13, "name": "hamster"}, {"supercategorie": "", "id": 14, "name": "horse"}, {"supercategorie": "", "id": 15, "name": "lion"}, {"supercategorie": "", "id": 16, "name": "lizard"}, {"supercategorie": "", "id": 17, "name": "monkey"}, {"supercategorie": "", "id": 18, "name": "motorcycle"}, {"supercategorie": "", "id": 19, "name": "rabbit"}, {"supercategorie": "", "id": 20, "name": "red_panda"}, {"supercategorie": "", "id": 21, "name": "sheep"}, {"supercategorie": "", "id": 22, "name": "snake"}, {"supercategorie": "", "id": 23, "name": "squirrel"}, {"supercategorie": "", "id": 24, "name": "tiger"}, {"supercategorie": "", "id": 25, "name": "train"}, {"supercategorie": "", "id": 26, "name": "turtle"}, {"supercategorie": "", "id": 27, "name": "watercraft"}, {"supercategorie": "", "id": 28, "name": "whale"}, {"supercategorie": "", "id": 29, "name": "zebra"}],
# }
#
# via_classes = ["bike", "bus", "car", "motor", "person", "truck", "traffic light",
#                "traffic sign", "delineator", "traffic light red", "traffic light green", "traffic light yellow",
#                "thermal_bus", "thermal_car", "thermal_truck", "thermal_person", "sl60",
#                "sl70", "sl80", "sl90", "sl100", "sl110", "stop", "sl25", "trolley"]
# res = []
# for idx, cls in enumerate(via_classes):
#     res.append({"supercategorie": "", "id": idx, "name": cls})
# print(res)

# photo to seq file

# Ovis = vid.OVIS(
#     data_dir="/media/hazelwang/FE92D56C92D529C3/yolov/YOLOV/dataset/aws/files/" ,
#     img_size=(1280,720),
#     COCO_anno= '/media/hazelwang/FE92D56C92D529C3/yolov/YOLOV/dataset/aws/files/20241119vid.json',
#     gframe= 16,
#     mode='random' # frame sample mode
#
# )
# Ovis.convert_ovis_coco(data_dir="/media/hazelwang/FE92D56C92D529C3/yolov/YOLOV/dataset/aws/files/" + '20241119vid.json',
#                        save_dir="/media/hazelwang/FE92D56C92D529C3/yolov/YOLOV/dataset/aws/files/" + '20241119ovis.json')