import torch
from yolox.exp import get_exp
from yolox.utils import fuse_model

# 导入模型权重
exp = get_exp('exps/default/yolox_s.py')
MyModel = exp.get_model()
ckpt = torch.load('YOLOX_outputs/yolov_s/best_ckpt.pth')#torch.load('./YOLOX_outputs/yolox_m/best_ckpt.pth',map_location='cpu')
MyModel.load_state_dict(ckpt['model'])
MyModel.eval()
MyModel = fuse_model(MyModel)
print(MyModel)
MyModel = torch.load('YOLOX_outputs/yolov_s/best_ckpt.pth')


# 输入数据格式
input_format = {
    'bs':1,
    'shape': (3,288, 512)
    }
demo = torch.randn((4,288,512,3))

# 输出onnx名字
export_onx_nm = '.demo/VIA/demo.onnx'
torch.onnx.export(
    MyModel,
    demo,
    opset_version=10,           # onnx 的版本
    do_constant_folding=True,	# 是否执行常量折叠优化
    input_names=["input"],		# 输入名
    output_names=["output"],	# 输出名
    dynamic_axes={"input":{0:"batch_size"},	# 批处理变量
                "output":{0:"batch_size"}}
)