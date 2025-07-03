import os
import cv2
from pathlib import Path
from PIL import Image
import os

def images_to_video(root_dir, output_path, fps=10, img_exts=('.jpg', '.png', '.jpeg')):
    """
    将多个文件夹下的图片合并为一个视频
    :param root_dir: 根目录，里面可以包含按天分类的子目录
    :param output_path: 输出视频路径，例如 output.mp4
    :param fps: 帧率
    :param img_exts: 支持的图片扩展名
    """
    print(f"正在从 {root_dir} 合并图片为视频：{output_path}")

    all_images = []

    for subdir, _, files in sorted(os.walk(root_dir)):
        files = sorted([f for f in files if f.lower().endswith(img_exts)])
        for fname in files:
            full_path = os.path.join(subdir, fname)
            all_images.append(full_path)

    if not all_images:
        print("未找到任何图片。请检查目录是否正确。")
        return

    # 读取第一张图片以获取宽高
    first_frame = cv2.imread(all_images[0])
    if first_frame is None:
        print("无法读取首帧图片。请检查图片是否损坏。")
        return

    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # .mp4 格式
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_path in all_images:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"跳过无法读取的图片: {img_path}")
            continue
        # 如果大小不同，可根据需要 resize：
        frame = cv2.resize(frame, (width, height))
        out.write(frame)

    out.release()
    print(f"视频保存成功：{output_path}")


def images_to_gif(image_folder, output_gif, duration=100, loop=0, img_exts=('.jpg', '.png', '.jpeg')):
    """
    将指定文件夹下的图片按顺序合成为 GIF 动画。

    :param image_folder: 图片文件夹路径
    :param output_gif: 生成的 GIF 路径，例如 'output.gif'
    :param duration: 每帧显示时间（毫秒）
    :param loop: 循环次数，0 表示无限循环
    :param img_exts: 支持的图片扩展名
    """
    images = sorted(
        [f for f in os.listdir(image_folder) if f.lower().endswith(img_exts)]
    )

    if not images:
        print("未找到任何图片。请确认路径和扩展名。")
        return

    frames = [Image.open(os.path.join(image_folder, img)) for img in images]

    # 保存为 GIF
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop
    )
    print(f"GIF 已生成：{output_gif}")

def resize_image(input_path, output_path, size):
    """
    Resize an image to the given size and save it.

    Args:
        input_path (str): Path to the input image file.
        output_path (str): Path where the resized image will be saved.
        size (tuple): New size as (width, height).
    """
    # 打开图片
    image = Image.open(input_path)

    # 调整大小
    size = (size[1], size[0])
    resized_image = image.resize(size, Image.ANTIALIAS)

    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存图像
    resized_image.save(output_path)
    print(f"Image saved to {output_path}")


if __name__ == "__main__":
    ILSVRC_path = "/media/hazelwang/C80ED71D0ED7037C/ILSVRC2015/Data/VID/test/ILSVRC2015_test_00001010/"
    test_path = '/media/hazelwang/FE92D56C92D529C3/yolov/YOLOV/dataset/VIA/ovis_format/val/Data/video000085/'
    res_path = '/media/hazelwang/FE92D56C92D529C3/yolov/YOLOV/YOLOX_outputs/yolov_s/vis_res/2025_05_28_09_56_40'

    # pics 2 video
    root_dir =   test_path
    output_video = root_dir+"output.mp4"
    images_to_video(root_dir, output_video, fps=10)

    # # pics 2 GIF
    # image_folder = "/media/hazelwang/FE92D56C92D529C3/yolov/YOLOV/YOLOX_outputs/yolov_s/vis_res/2025_05_28_09_56_40/"  # 例如 "dataset/images/"
    # output_gif = image_folder + "output.gif"
    # images_to_gif(image_folder, output_gif, duration=40)

    # resize the pic
    # input_file = "dataset/VIA/coco_format/val/Data/video000001/image000000.JPEG"
    # output_file = "demo/VIA/test.JPEG"
    # new_size = (288, 512)  # height, width
    #
    # resize_image(input_file, output_file, new_size)
