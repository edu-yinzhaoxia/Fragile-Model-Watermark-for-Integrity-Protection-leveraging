import math
import dill

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import resnet20
import torch.nn.init as init
import torchvision.models as models

def get_last_layer(model):
    model_name = model.__class__.__name__

    if model_name.startswith("ResNet"):
        return model.fc   #  or fc层(resnet18系列)
    elif model_name == "AlexNet":
        return model.classifier[-1]
    elif model_name == "LeNet":  # 假设你的LeNet的最后一层命名为'fc3'
        return model.fc3
    elif 'vgg' in model_name.lower():
        return model.classifier[-1]
    elif 'visiontransformer' in model_name.lower():  # 根据实际情况修改，确保模型名称包含 'visiontransformer'
        # 根据 ViT 模型的结构或文档，指定正确的最后一层
        return model.heads.head  # 这里替换为 ViT 模型的实际最后一层
    elif 'shuffle' in model_name.lower():
        return model.fc
    else:
            raise ValueError(f"Unsupported model: {model_name}")


class ExtendedModel(nn.Module):
    def __init__(self, base_model, num_classes,init_method='kaiming_uniform'):
        """"
        初始化函数

        param base_model: 外部输入的模型
        param num_classes: 输出类别数量
        """
        super(ExtendedModel, self).__init__()

        # 保留原始模型
        self.base_model = base_model

        # 获取原始模型的最后一层输出特征数
        # 这里假设base_model的最后一层是nn.Linear类型，如果不是这样，请根据实际情况修改
        last_layer = get_last_layer(base_model)
        num_features = last_layer.out_features

        # 在原始模型的基础上添加一个新的线性层
        self.new_classifier_1 = nn.Linear(num_features, num_classes)
        self.activity = nn.ReLU()
        self.new_classifier_2 = nn.Linear(num_features,num_classes)
        # 应用初始化方法
        if init_method == 'xavier_uniform':
            init.xavier_uniform_(self.new_classifier_1.weight)
        elif init_method == 'xavier_normal':
            init.xavier_normal_(self.new_classifier_1.weight)
        elif init_method == 'kaiming_uniform':
            init.kaiming_uniform_(self.new_classifier_1.weight, nonlinearity='relu')
        elif init_method == 'kaiming_normal':
            init.kaiming_normal_(self.new_classifier_1.weight, nonlinearity='relu')
        elif init_method == 'zeros':
            init.zeros_(self.new_classifier_1.weight)
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

    def forward(self, x):
        """
        前向传播函数

        :param x: 输入数据
        :return: 线性层的输出
        """
        # 通过原始模型
        y = self.base_model(x)
        # y = self.new_classifier_1(y)
        # y = self.activity(y)
        # 通过新增的线性层
        output = self.new_classifier_1(y)

        return output

if __name__ == '__main__':


    save_root = ''



    model = torch.load('')
    #model = torch.load(load_root)
    oldModel = model
    print(oldModel)



    for param in oldModel.parameters():
        param.requires_grad = False
    newModel = ExtendedModel(oldModel, 2 ,init_method='kaiming_uniform')
    #torch.save(newModel.state_dict(), save_root)
    torch.save(newModel, save_root)
    #torch.save(newModel,save_root,pickle_module=dill)


    print(newModel)
