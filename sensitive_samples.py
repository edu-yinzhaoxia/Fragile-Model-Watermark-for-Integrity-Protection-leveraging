import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from addLayer import ExtendedModel
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import time
from torchvision import datasets, transforms
from torch.nn import DataParallel

from torch.utils.data import DataLoader
from AID import ActivationHook
from addLayer import ExtendedModel
import dill
import resnet20

model = torch.load('')

model = model.to('cuda')  # Move the model to GPU
model.eval()  # Set to evaluation mode

input_data = torch.randn(300, 3, 32, 32, requires_grad=True)# Keep input_data as a leaf tensor
input_data_gpu = input_data.to('cuda')  # Use a GPU copy for operations
optimizer = optim.SGD([input_data], lr=1e-3)  # Optimize the original tensor

transform = transforms.Compose([
    transforms.ToTensor(),
])




start_time = time.time()

for epoch in range(1001):  # For demonstration, we only run 10 epochs
    optimizer.zero_grad()

    # for data, target in train_loader:
    #     data = data.to('cuda')  # Move batch of images to GPU
    #     break

    """
    """
    # hook = ActivationHook()
    # hook.register_hook(model)
    """
    """

    input_data_gpu = input_data.to('cuda')
    output = model(input_data_gpu)
    #output_softmax = F.softmax(output, dim=1)  # 通过softmax获得概率分布
    num_classes = output.size(1)  # 获取类别的数量


    variance_reduction_loss = torch.var(output,dim=1)  # Minimize the variance of the output logits
    variance_reduction_loss = torch.mean(variance_reduction_loss)
#####

   # variance_reduction_loss = torch.pow(variance_reduction_loss, 1)  # where p > 1 to amplify larger variances more

    # activation_loss

    # Combined loss
    combined_loss =   100 * variance_reduction_loss
    """
    """
    # activation_loss = hook.compute_loss()
    # aid_loss = variance_reduction_loss  + (0.25) * activation_loss
    """
    """
    # Backward pass
    combined_loss.backward()


    # Update the input data
    optimizer.step()
    input_data.data = torch.clamp(input_data.data, min=0, max=1)
    print(epoch)


    if epoch % 5000 == 0 :
        # 假设 output 是您的模型输出
        output_cpu = output.cpu().detach().numpy().flatten()
        probs = F.softmax(output, dim=1)

        # 获取 output 的长度
        probs_cpu = probs.cpu().detach().numpy().flatten()

        # 获取 probs 的长度
        probs_length = len(probs_cpu)

        # 创建标签的数组（0, 1, 2, ..., probs_length-1）
        labels = np.arange(probs_length)

        # 创建条形图
        plt.bar(labels, probs_cpu,)
        #plt.xticks(np.arange(0, probs_length, 1))

        default_blue = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
        colors = [default_blue] * probs_length

        max_value = input_data.max().item()
        min_value = input_data.min().item()
        mean_value = input_data.mean().item()
        output = output.sum().item()/input_data.size(0)
        print(f"Input Data - Max: {max_value}, Min: {min_value}, Mean: {mean_value}, Output: {output}")

        print(f"Epoch [{epoch }/5000], Combined Loss: {combined_loss.item()}")
    #torch.save(input_data_gpu,'./sensitive_samples/resnet152/flowers102/PureVar_resnet152_flowers102_300.pt')
end_time = time.time()
elapsed_time = end_time - start_time
print(f"The operation took {elapsed_time:.2f} seconds.")
