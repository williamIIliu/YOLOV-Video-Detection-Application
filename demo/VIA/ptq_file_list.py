# from yolox.data.datasets import vid
# from yolox.data.data_augment import Vid_Val_Transform,ValTransform
# import argparse
#
# def make_parser():
#     parser = argparse.ArgumentParser("PTQ file list maker")
#     parser.add_argument(
#         "-p",
#         "--path",
#         type=str,
#         default='seq file npy',
#         help="Path to your input image.",
#     )
#     parser.add_argument(
#         "-o",
#         "--output_path",
#         type=str,
#         default='demo_output',
#         help="Path to your output directory.",
#     )
#     parser.add_argument(
#         "--test_size",
#         type=str,
#         default="640,640",
#         help="Specify an test shape for inference.",
#     )
#     return parser
#
# def ptq_file_maker(file_path, test_size, txt_path):
#     dataset = vid.VIDDataset(
#         file_path=file_path,
#         img_size=test_size,
#         preproc=ValTransform(
#                              legacy=False
#                          )
#         )
#     dataset = vid.get_trans_loader(batch_size=1, data_num_workers=4, dataset=dataset)
#     # tmp = dataset.__iter__()
#     # print(tmp)
#     for idx, img, target, img_info, path in enumerate(dataset):
#         if idx > 2:
#             break
#         print(path)
#
#
#     return dataset
#
# if __name__ == '__main__':
#     args = make_parser().parse_args()
#     test_shape = tuple(map(int, args.test_size.split(',')))
#
#     dataset = ptq_file_maker(
#         args.path,
#         test_shape,
#         args.output_path,
#     )
#     print(dataset)


import numpy as np
import cv2
import os

data_path = np.load('./dataset/VIA/ovis_format/val_seq.npy',allow_pickle=True)
print(data_path)
data_list = data_path.tolist()
# print(data_list)
# print(len(data_list))

img_list = []
for vid in data_list:
    for pic in vid:
        img_list.append('/media/hazelwang/FE92D56C92D529C3/yolov/YOLOV'+pic[1:])

# print(img_list)
print(len(img_list))


out_dir = "/media/hazelwang/FE92D56C92D529C3/yolov/YOLOV/demo/VIA/SNPEImgs/"
os.makedirs("./demo/VIA/SNPEImgs", exist_ok=True)
file_list = open("./demo/VIA/SNPEImgs/SNPEImgList.txt", 'w')
# flist = open("SNPEImgList_512_288.txt", 'w')

for idx, img_path in enumerate(img_list):
    if idx >= 100:
        break
    img = cv2.imread(img_path)
    print(img.shape)
    img = cv2.resize(img, (512, 288)) #(width, height)
    print(img.shape)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img)
    # img[:, :, 0] = ((img[:, :, 0] - 104.01362025) / 73.60276650)  # / 127.5 - 1.0
    # img[:, :, 1] = ((img[:, :, 1] - 114.03422265) / 69.89082075)  # / 127.5 - 1.0
    # img[:, :, 2] = ((img[:, :, 2] - 119.91659580) / 70.91507670)  # / 127.5 - 1.0

    img_name = img_path.split("/")[-1].replace(".JPEG", ".data")
    print(img_name)

    file_list.write(out_dir + img_name + "\n")
    f = open(out_dir + img_name, 'wb')
    f.write(img)
    f.close()

file_list.close()