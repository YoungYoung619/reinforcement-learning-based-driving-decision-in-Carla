"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""

import torch
import torch.nn as nn
import torchvision as tv
import time

from agents.DQN_agents.base_conv_net.mobilenet_v2 import MobileNetV2


class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        return x


def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean) / var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels,
                           conv.out_channels,
                           conv.kernel_size,
                           conv.stride,
                           conv.padding,
                           groups=conv.groups,
                           bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv


def fuse_module(m):
    children = list(m.named_children())
    c = None
    cn = None

    for name, child in children:
        if isinstance(child, nn.BatchNorm2d):
            bc = fuse(c, child)
            m._modules[cn] = bc
            m._modules[name] = DummyModule()
            c = None
        elif isinstance(child, nn.Conv2d):
            c = child
            cn = name
        else:
            fuse_module(child)


def test_net(m):
    p = torch.randn([1, 3, 43, 300])
    s = time.time()
    o_output = m(p)
    print("Original time: ", time.time() - s)
    # print(o_output)
    fuse_module(m)

    s = time.time()
    f_output = m(p)
    # print(f_output)
    print("Fused time: ", time.time() - s)

    print("Max abs diff: ", (o_output - f_output).abs().max().item())
    assert (o_output.argmax() == f_output.argmax())
    # print(o_output[0][0].item(), f_output[0][0].item())
    print("MSE diff: ", nn.MSELoss()(o_output, f_output).item())


def test_layer():
    p = torch.randn([1, 3, 112, 112])
    conv1 = m.conv1
    bn1 = m.bn1
    o_output = bn1(conv1(p))
    fusion = fuse(conv1, bn1)
    f_output = fusion(p)
    print(o_output[0][0][0][0].item())
    print(f_output[0][0][0][0].item())
    print("Max abs diff: ", (o_output - f_output).abs().max().item())
    print("MSE diff: ", nn.MSELoss()(o_output, f_output).item())

def load_pretrain_model(net, model):
    net_dict = net.state_dict()
    if not torch.cuda.is_available():
        pretrain_dict = torch.load(model, map_location='cpu')
    else:
        pretrain_dict = torch.load(model)
    # print(net_dict.keys())
    # print(pretrain_dict.keys())
    load_dict = {(k): v for k, v in pretrain_dict.items() if
                 (k) in net_dict}
    print(load_dict.keys())
    net_dict.update(load_dict)



if __name__ == "__main__":
    # m = tv.models.resnet152(False)
    # print(m)
    # m.eval()
    # print("Layer level test: ")
    # test_layer()
    model_path = 'E:\\reinforcement-learning-based-driving-decision-in-Carla\\agents\DQN_agents\\base_conv_net\pretrain\mobilenetv2_0.35-b2e15951.pth'
    # m = MobileNetV2(n_class=244, width_mult=0.5)

    #### method-1: load dict
    m = MobileNetV2(n_dim=128)
    # load_pretrain_model(m, model_path)

    fuse_module(m)
    print(m)
    m.load_state_dict(torch.load('./pretrain/mobilenetv2_0.35-b2e15951_nobn.pth'), strict=True)
    # torch.save(m.state_dict(), './pretrain/mobilenetv2_0.35-b2e15951_nobn.pth')


# if __name__ == '__main__':
#     inputs = torch.rand(5, 3, 224, 224)
#     net = MobileNetV2(n_dim=128)
#     dd = net.state_dict()
#     net_keys = list(dd.keys())
#     net_vals = list(dd.values())
#     out = net(inputs)
#
#     pretarin_model = 'E:\\reinforcement-learning-based-driving-decision-in-Carla\\agents\DQN_agents\\base_conv_net\pretrain\mobilenetv2_0.35-b2e15951.pth'
#     pretrain_dict = torch.load(pretarin_model)
#     pretrain_keys = list(pretrain_dict.keys())
#     pretrain_vals = list(pretrain_dict.values())
#
#     idx = 0
#     for pretrain_key in pretrain_keys:
#         if len(pretrain_dict[pretrain_key].size()) == 4 and list(pretrain_dict[pretrain_key].size()) == list(net_vals[idx].size()):
#             dd[net_keys[idx]] = pretrain_dict[pretrain_key]
#             print('%s load success'%(net_keys[idx]))
#             idx += 1
#
#     net.load_state_dict(dd, strict=True)
#     out1 = net(inputs)
#     torch.save(net.state_dict(), './pretrain/mobilenetv2_0.35-b2e15951_no_bn.pth')







