ckpt_path = '/YOLOV/ckpt/v++_large/last_epoch_ckpt/data.pkl'
files = '/YOLOV/exps/yolov++/v++_large_decoupleReg.py'
video = '/YOLOV/dataset/demo/hess-20250327-2.mp4'
#python tools/vid_demo.py -f [path to your yolov exp files] -c 'YOLOV/ckpt/v++_large.pth' --path /path/to/your/video --conf 0.25 --nms 0.5 --tsize 576 --save_result
python ./tools/vid_demo.py -f './exps/yolov/yolov_s.py' -c './ckpt/yolov_s.pth' --path './dataset/demo/hess-20250327-2.mp4' --conf 0.25 --nms 0.5 --tsize 576