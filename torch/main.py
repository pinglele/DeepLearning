'''from pprint import pprint

model_names = timm.list_models(pretrained=True)
print("支持的预训练模型数量：%s" % len(model_names))

strs = '*resne*t*'
model_names = timm.list_models(strs)
print("通过通配符 %s 查询到的可用模型：%s" % (strs, len(model_names)))
mobilenetv2_035
model_names = timm.list_models(strs, pretrained=True)
print("通过通配符 %s 查询到的可用预训练模型：%s" % (strs, len(model_names))
available_models = timm.list_models()
for model_name in available_models:
    if model_name.startswith('e'):
        print(model_name)'''
#
# '''print("如果不设置num_classes，表示使用的是原始的预训练模型的分类层")
# m
# m.eval()
# o = m(torch.randn(2, 3, 224, 224))
# print(f'Classification layer shape: {o.shape}')
# #输出flatten层或者global_pool层的前一层的数据（flatten层和global_pool层通常接分类层）
# o = m.forward_features(torch.randn(2, 3, 224, 224))
# print(f'Feature shape: {o.shape}\n')'''
# #timm的基本使用
'''import torch,timm
import torchvision
from torchstat import stat
from timm.models import create_model

print(torch.cuda.is_available())  #注意是双下划线

available_models = timm.list_models()
for model_name in available_models:
    if model_name.startswith('e'):
        print(model_name)
model = timm.create_model('efficientnet_b6', pretrained=False)
device = torch.device('cuda:0')
model = model.to(device)
stat(model, (3, 224, 224))
/*****************所用网络***************/
mobilenetv2_100     vgg16       resnet18       mobilenetv3_small_100    
mobilenetv3_large_100      mobilevitv2_100     efficientnet_b5      efficientnet_b6     efficientnet_b7
efficientnetv2_rw_s     efficientnetv2_l
'''
import torch

from torchsummary import summary

from timm import create_model

from torchvision import transforms


# 创建模型

model = create_model('efficientnet_b0', pretrained=False)


# 设定输入尺寸

input_shape = (3, 224, 224)


# 判断是否有GPU可用，选择设备

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)


# 传递设备类型的字符串到summary函数

summary(model, input_shape, device=device.type)

