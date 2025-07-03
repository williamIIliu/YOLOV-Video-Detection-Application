'''
1)run convert_ckpt_to_hef.py
->centernet_optimized.pb
2)pb to onnx command:
python -m tf2onnx.convert --input centernet_optimize.pb --inputs input_tensor:0[1,288,512,3] --inputs-as-nchw input_tensor:0 --rename-inputs inputs --output con_ok_12.onnx --outputs Identity:0,Identity_1:0,Identity_2:0,Identity_3:0,Identity_4:0,Identity_5:0,Identity_6:0 --outputs-as-nchw Identity:0,Identity_1:0,Identity_2:0,Identity_3:0,Identity_4:0,Identity_5:0,Identity_6:0 --opset 12
->model.onnx
3)run modify_onnx.py
->edited_model.onnx
4)parse to har. (hailo_venv)
hailomz parse model_name
note: change model_name from hailo_model_zoo/hailo_model_zoo/cfg/networks/model_name.yaml
5)run convert_ckpt_to_hef.py with flags all False
->.hef
'''
import onnx
from onnx import helper, TensorProto
import numpy as np
import onnx
from onnxsim import simplify


work_dir = "/home/hazelwang/PycharmProjects/data/object_detection_calib_test_data/BlueDot_v0.1.5.49_20240528/ckpt/frozen_models/"


def update_resize_output_and_remove_node(onnx_path):
    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph
    resize_op_names = ['Resize__249', 'Resize__257', 'Resize__265'] # for centernet model
    # resize_op_names = ['Resize__133', 'Resize__141', 'Resize__149'] # for lane model
    for node in graph.node:
        if node.name in resize_op_names:
            for att in node.attribute:
                if att.name == "nearest_mode":
                    print("Ted is hereB")
                    att.s = b'round_prefer_floor'
    
    onnx_model.graph.output[0].name = 'model_1/conv2d_7/BiasAdd:0'
    onnx_model.graph.output[1].name = 'model_1/conv2d_8/BiasAdd:0'
    onnx_model.graph.output[2].name = 'model_1/conv2d_9/BiasAdd:0'
    onnx_model.graph.output[3].name = 'Version/bias:0'

    delete_op_names = ['Squeeze'] # for heatmap
    for node in graph.node:
        if node.name in delete_op_names:
            graph = delete_node(node, graph)

    delete_op_names = ['Cast/x'] # for heatmap
    for node in graph.node:
        if node.name in delete_op_names:
            graph = delete_node(node, graph)


    delete_op_names = ['Squeeze_1'] # for scale
    for node in graph.node:
        if node.name in delete_op_names:
            graph = delete_node(node, graph)

    delete_op_names = ['Cast_1/x'] # for scale
    for node in graph.node:
        if node.name in delete_op_names:
            graph = delete_node(node, graph)


    delete_op_names = ['Squeeze_2'] # for offset
    for node in graph.node:
        if node.name in delete_op_names:
            graph = delete_node(node, graph)

    delete_op_names = ['Cast_2/x'] # for offset
    for node in graph.node:
        if node.name in delete_op_names:
            graph = delete_node(node, graph)


    delete_op_names = ['Identity_version_graph_outputs_Identity__11__16'] # for version
    for node in graph.node:
        if node.name in delete_op_names:
            graph = delete_node(node, graph)


    delete_op_names = ['Squeeze_version'] # for version
    for node in graph.node:
        if node.name in delete_op_names:
            graph = delete_node(node, graph)

    delete_op_names = ['Cast_version/x'] # for version
    for node in graph.node:
        if node.name in delete_op_names:
            graph = delete_node(node, graph)

    delete_op_names = ['Version/bias__298'] # for version
    for node in graph.node:
        if node.name in delete_op_names:
            graph = delete_node(node, graph)


    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, work_dir + "centernet_optimize_v2_refine.onnx")


def replace_resize_node_to_convtranspose(onnx_path):
    model = onnx.load(onnx_path)
    node = model.graph.node
    # 搜索目标节点
    # for i in range(len(node)):
    #     if node[i].op_type == 'Resize':
    #         node_rise = node[i]
    #         print(i)
    #         # print(node_rise)

    # 节点shape
    resize_128_shape = [512, 512, 2, 2]
    resize_136_shape = [256, 256, 2, 2]
    # resize_144_shape = [64, 64, 2, 2]

    # 生成weight
    weight1 = np.zeros([512, 512, 2, 2]).astype(np.float32)
    weight2 = np.zeros([256, 256, 2, 2]).astype(np.float32)
    # weight3 = np.zeros([64, 64, 2, 2]).astype(np.float32)

    # 初始化权值
    for i in range(512):
        weight1[i, i, :, :] = 1

    for i in range(256):
        weight2[i, i, :, :] = 1

    # for i in range(64):
    #     weight3[i, i, :, :] = 1


    # define weight
    deconv1_weight = helper.make_tensor('deconv1.weight', TensorProto.FLOAT, resize_128_shape, weight1)
    deconv2_weight = helper.make_tensor('deconv2.weight', TensorProto.FLOAT, resize_136_shape, weight2)
    # deconv3_weight = helper.make_tensor('deconv3.weight', TensorProto.FLOAT, resize_144_shape, weight3)

    # 新建新节点
    deconv1_node = onnx.helper.make_node(
    name="ConvTranspose_165",
    op_type="ConvTranspose",
    inputs=["662", 'deconv1.weight'],  # name: model_2/conv2d/BiasAdd:0
    outputs=["667"],  # name: Resize__133:0
    output_padding=[0, 0],
    group=1,
    kernel_shape=[2, 2],
    pads=[0, 0, 0, 0],
    strides=[2, 2],
    dilations=[1, 1]
    )

    # 新建新节点
    deconv2_node = onnx.helper.make_node(
    name="ConvTranspose_187",
    op_type="ConvTranspose",
    inputs=["693", 'deconv2.weight'],  # name: model_2/re_lu/Relu6:0
    outputs=["698"],  # name: Resize__141:0
    output_padding=[0, 0],
    group=1,
    kernel_shape=[2, 2],
    pads=[0, 0, 0, 0],
    strides=[2, 2],
    dilations=[1, 1]
    )

    # 新建新节点
    # deconv3_node = onnx.helper.make_node(
    # name="ConvTranspose_149",
    # op_type="ConvTranspose",
    # inputs=["model_2/re_lu_1/Relu6:0", 'deconv3.weight'],  # name: model_2/re_lu_1/Relu6:0
    # outputs=["Resize__149:0"],  # name: Resize__149:0
    # output_padding=[0, 0],
    # group=1,
    # kernel_shape=[2, 2],
    # pads=[0, 0, 0, 0],
    # strides=[2, 2],
    # dilations=[1, 1]
    # )


    # remove Resize node
    model.graph.node.remove(node[128])

    # # add ConvTranspose node
    model.graph.node.insert(128, deconv1_node)

    # # init weight of ConvTranspose node
    model.graph.initializer.append(deconv1_weight)


    # remove Resize node
    model.graph.node.remove(node[149])

    # # add ConvTranspose node
    model.graph.node.insert(149, deconv2_node)

    # # init weight of ConvTranspose node
    model.graph.initializer.append(deconv2_weight)


    # # remove Resize node
    # model.graph.node.remove(node[156])
    #
    # # # add ConvTranspose node
    # model.graph.node.insert(156, deconv3_node)
    #
    # # # init weight of ConvTranspose node
    # model.graph.initializer.append(deconv3_weight)

    #for output in model.graph.output:
    #    output.type.tensor_type.elem_type = TensorProto.FLOAT16

    # 模型检查
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)
    else:
        print('The model is valid!')

    # # save model
    # onnx.save(model, work_dir + "con_ok_12_final.onnx")
    onnx.save(model, "/home/hazelwang/PycharmProjects/data/william/yolox_s_modifyresize.onnx")


'''
for reference, no use in this script
'''
def createGraphMemberMap(graph_member_list):
    member_map = {}
    for n in graph_member_list:
        member_map[n.name] = n
    return member_map
def delete_node(node_name, graph):
    graph.node.remove(node_name)
    return graph
def modify_node(node_name, graph):
    for input_node in graph.input:
        if 'input_xxx' == input_node.name:
            print("change input data name")
            input_node.name = 'data'
def add_node(graph):
    '''example is -mean, /std'''
    '''new a tensor which is the value needs to be minus'''
    sub_const_node = onnx.helper.make_tensor(name='const_sub',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=[1],
                                             vals=[-127.5])
    '''add tensor to the graph'''
    graph.initializer.append(sub_const_node)
    '''new a node and add it to the graph'''
    sub_node = onnx.helper.make_node('Add',
                                     name='pre_sub',
                                     inputs=['data','const_sub'],
                                     outputs=['pre_sub'])
    graph.node.insert(0, sub_node)
    '''same as mul'''
    mul_const_node = onnx.helper.make_tensor(name='const_mul',
                                             data_type=onnx.TensorProto.FLOAT,
                                             dims=[1],
                                             vals=[1.0/127.5])
    graph.initializer.append(mul_const_node)
    sub_node = onnx.helper.make_node('Mul',
                                     name='pre_mul',
                                     inputs=['pre_sub', 'const_mul'],
                                     outputs=['pre_mul'])
    graph.node.insert(1, sub_node)
    '''modify the first conv layer'''
    for id, node in enumerate(graph.node):
        for i, input_node in enumerate(node.input):
            if 'data' == input_node:
                node.input[i] = 'pre_mul'

if __name__=='__main__':
    # # load your predefined ONNX model
    # model = onnx.load('/home/hazelwang/PycharmProjects/data/object_detection_calib_test_data/BlueDot_v0.1.5.49_20240528/ckpt/frozen_models/con_ok_12.onnx') # step3
    #
    # # convert model
    # model_simp, check = simplify(model)
    #
    # assert check, "Simplified ONNX model could not be validated"

    # onnx_path = work_dir + "con_ok_12_simplify.onnx"

    # onnx.save(model_simp, onnx_path)

    onnx_path = '/home/hazelwang/PycharmProjects/data/william/yolox_s.onnx'
    # model = onnx.load(onnx_path)
    # node = model.graph.node
    # # 搜索目标节点
    # for i in range(len(node)):
    #     if node[i].op_type == 'Resize':
    #         node_rise = node[i]
    #         print(i)
    #         print(node_rise)

    replace_resize_node_to_convtranspose(onnx_path) # step2

