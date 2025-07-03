import cv2
import numpy as np

# IoU Loss
# img = np.zeros((512, 512, 3), np.uint8)
# img.fill(255)
#
# # 分别是矩形左上、右下的坐标
# RecA = [30, 30, 300, 300]
# RecB = [60, 60, 350, 340]
#
# cv2.rectangle(img, (RecA[0], RecA[1]), (RecA[2], RecA[3]), (0, 255, 0), 5)
# cv2.rectangle(img, (RecB[0], RecB[1]), (RecB[2], RecB[3]), (255, 0, 0), 5)
#
#
# def IoULoss(rec1, rec2, eps=1e-6):
#     '''
#     rec: [x_min, y_min, x_max, y_max]
#     '''
#     # 求交叉部分坐标，算的是占据像素点而非边长，所以加1
#     x_i_min = max(rec1[0], rec2[0])
#     y_i_min = max(rec1[1], rec2[1])
#     x_i_max = min(rec1[2], rec2[2])
#     y_i_max = min(rec1[3], rec2[3])
#     A_int = max(0, x_i_max - x_i_min + 1) * max(0, y_i_max - y_i_min + 1)
#
#     # 公共
#     A_uni = (rec1[2] - rec1[0] + 1) * (rec1[3] - rec1[1] + 1) \
#             + (rec2[2] - rec2[0] + 1) * (rec2[3] - rec2[1] + 1) \
#             - A_int
#
#     return  -1 * np.log(A_int / A_uni + eps)
#
# iou_los = IoULoss(RecA, RecB)
# print(iou_los)
#
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img,"IOU = %.2f"%iou_los,(130, 190),font,0.8,(0,0,0),2)
# cv2.imshow("image",img)
# cv2.waitKey()
# cv2.destroyAllWindows()


# NMS
# def py_cpu_nms(dets, thresh):
#     """Pure Python NMS baseline."""
#     # x1、y1、x2、y2、以及score赋值
#     x1 = dets[:, 0]
#     y1 = dets[:, 1]
#     x2 = dets[:, 2]
#     y2 = dets[:, 3]
#     recs = dets[:,:4]
#     scores = dets[:, 4]
#
#     # 每一个检测框的面积
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     # 按照score置信度降序排序
#     order = scores.argsort()[::-1]
#
#     keep = []  # 保留的结果框集合
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)  # 保留该类剩余box中得分最高的一个
#         # 得到相交区域,左上及右下
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])
#
#         # 计算相交的面积,不重叠时面积为0
#         w = np.maximum(0.0, xx2 - xx1 + 1)
#         h = np.maximum(0.0, yy2 - yy1 + 1)
#         inter = w * h
#         # 计算IoU：重叠面积 /（面积1+面积2-重叠面积）
#         ovr = inter / (areas[i] + areas[order[1:]] - inter)
#         # 保留IoU小于阈值的box
#         inds = np.where(ovr <= thresh)[0]
#         order = order[inds + 1]  # 因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位
#
#     return keep


# model quantumlization
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.ao.quantization as quant
#
# class model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.quant = quant.QuantStub()
#         self.linear1 = nn.Linear(3,3,bias=False)
#         self.relu = nn.ReLU()
#         self.linear2 = nn.Linear(3,1,bias=False)
#         self.dequant = quant.DeQuantStub()
#
#     def forward(self,x):
#         q_inputs = self.quant(x)
#         logits1 = self.linear1(q_inputs)
#         activation1 = self.relu(logits1)
#         yhat = self.linear2(activation1)
#         q_yhat = self.dequant(yhat)
#         return q_yhat
#
# # 构造数据
# weight = torch.tensor(
#     [
#         [1.1],
#         [2.1],
#         [3.1]
#     ]
# )
# X_train = torch.randn(10000,3)
# Y_train = X_train @ weight
# X_test = torch.randn(1000,3)
# Y_test = X_test @ weight
#
# # 训练数据
# MyModel = model()
# optimizer = torch.optim.SGD(
#     MyModel.parameters(),
#     lr=1e-3,
#     momentum=0.98
# )
# for i in range(100):
#     yhat = MyModel(X_train)
#     loss = F.mse_loss(yhat,Y_train)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#
# # 训练完之后，用于推理，分别使用原始结果和量化结果
# MyModel.eval()
# with torch.no_grad():
#     preds = MyModel(X_test)
#     loss_test = F.mse_loss(preds, Y_test)
#     print(f'MSE loss in the float32 model:{loss_test.item():.3f}')

# torch自带量化函数，动态量化模型
# MyModel_int8 = torch.ao.quantization.quantize_dynamic(
#     MyModel,
#     {nn.Linear},
#     dtype = torch.qint8
# )
# with torch.no_grad():
#     preds = MyModel(X_test)
#     loss_test = F.mse_loss(preds, Y_test)
#     print(f'MSE loss in the float32 model:{loss_test.item():.3f}')
#
# print('float32 model linear1 weight params:\n',MyModel.linear1.weight)
# print('int8 model linear1 weight params (int8):\n',torch.int_repr(MyModel_int8.linear1.weight()))
# print('int8 model linear1 weight params (float32):\n',MyModel_int8.linear1.weight())

# torch自带量化函数，静态量化模型
# MyModel.qconfig = quant.get_default_qconfig('x86')
# MyModel_fused = quant.fuse_modules(
#     MyModel,
#     [['linear1', 'relu']]
# )
# MyModel_prepared = quant.prepare(MyModel)
# MyModel_prepared(X_test)
#
# MyModel_int8 = quant.convert(MyModel_prepared)
# with torch.no_grad():
#     preds = MyModel(X_test)
#     loss_test = F.mse_loss(preds, Y_test)
#     print(f'MSE loss in the int8 model:{loss_test.item():.3f}')
#
# print('float32 model linear1 weight params:\n',MyModel.linear1.weight)
# print('int8 model linear1 weight params (int8):\n',torch.int_repr(MyModel_int8.linear1.weight()))
# print('int8 model linear1 weight params (float32):\n',MyModel_int8.linear1.weight())



# 反卷积代替最近邻插值
import torch
import torch.nn as nn

x = torch.tensor([[[[1., 2., 3., 4.],
                    [3., 4., 5. ,6.]]]])

# 设置 TransposeConv2d，kernel=2, stride=2，权重全为 1，bias=0
deconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
deconv.weight.data[:] = 1.0

print(deconv(x))
