# 定义 FGSM 攻击函数

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from addLayer import ExtendedModel
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import resnet20

input_data = torch.load('')

model = torch.load('')

model = model.to('cuda:2')  # Move the model to GPU
model.eval()
# Set to evaluation mode
# print(model)

reached_boundary = torch.zeros(input_data.size(0), dtype=torch.bool, device='cuda:2')


def visualize_probs(output):
    probs = F.softmax(output, dim=1)
    probs_cpu = probs.cpu().detach().numpy().flatten()
    labels = np.arange(len(probs_cpu))

    plt.figure(figsize=(10,5))
    bars = plt.bar(labels, probs_cpu)

    # 在每个条形上添加文本
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 6), ha='center', va='bottom')

    plt.xlabel('Class Labels')
    plt.ylabel('Probabilities')
    plt.title('Probabilities by Class Label')
    plt.show()

# 在PGD攻击中使用此函数：
def pgd_attack_viz(model, image, alpha, iters):
    original_image = image.clone()
    successful_attack_image = torch.empty_like(image).to('cuda:2')
    almost_successful_attack_image = torch.empty_like(image).to('cuda:2')

    for i in range(iters):

        image.requires_grad = True
        image.to('cuda:2')

        prev_image = image.clone().to('cuda:2')

        if i != 0:
            last_time_outputs = outputs.clone()


        outputs = model(image.to('cuda:2'))
        #print(outputs.sum(dim=1))
        if i == 0:
            labels = torch.max(outputs, 1)[1]   #当前标签情况
        model.zero_grad()
        # 计算损失
        # loss = F.cross_entropy(output, label)
        loss = F.cross_entropy(outputs[~reached_boundary], labels[~reached_boundary]) #第一次训练的时候肯定是所有样本都要训练的

        loss.backward()
        data_grad = image.grad.data

        # 执行攻击
        perturbed_image = image + alpha * data_grad.sign()
        image = torch.clamp(perturbed_image, min=0, max=1).detach_()

        # 检查哪些图像的标签已经改变

        changed_labels = (labels != torch.max(model(image.to('cuda:2')), 1)[1])


        changed_labels_this_iter = changed_labels & ~reached_boundary #reached_boundary用于记录上一轮到达边界的情况，changed_labels用于记录本轮到达边界的情况
        almost_successful_attack_image[changed_labels_this_iter] = prev_image[changed_labels_this_iter]
        reached_boundary[changed_labels_this_iter] = True

        successful_attack_image[reached_boundary] = image.to('cuda:2')[reached_boundary]

        #image.grad.data.zero_()
        torch.cuda.empty_cache()

        # 可视化概率
        # if i % 45 == 0:
        #     visualize_probs(outputs)

        changed_classification = labels != torch.max(model(image.to('cuda:2')), 1)[1]
        if torch.any(changed_labels):
            #print(i)
            torch.save(almost_successful_attack_image,'./sensitive_samples/resnet152/flowers102/almost_successful_attack_image_bj_150.pt')
            torch.save(successful_attack_image, './sensitive_samples/resnet152/flowers102/successful_attack_image_bj_150.pt')
        if torch.any(changed_labels_this_iter):
            #print(torch.max(model(image.to('cuda')), 1)[1])
            print(changed_labels.tolist().count(True))
            # visualize_probs(outputs)
            # visualize_probs(model(image.to('cuda')))
        if torch.all(reached_boundary):
            print(i)
            break


pgd_attack_viz (model, input_data, 1e-6, 10000)