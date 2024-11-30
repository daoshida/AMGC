from AMGC_ResNet18 import SKConv, Attention, Conv2d_bn, RFB, ResNet, ResBlock
import random
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import argparse
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


class SimpleDateset(Dataset):
    def __init__(self,data_file,transform):
        with open(data_file,'r') as f:
            self.meta = json.load(f)
        self.transform_d = transform
    def __getitem__(self,i):
        data_path = Path(self.meta['data_path'][i])
        data = np.load(data_path,allow_pickle=True)
        data = self.transform_d(data)
        data = data.permute(1,2,0)
#         data = data.unsqueeze(0)
#         data = data.squeeze(0)
        data = data.to(torch.float32)
        label = int(self.meta['data_labels'][i])
        return data,label
    def __len__(self):
        return len(self.meta['data_labels'])
    

# 参数设定
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
parser.add_argument('--max_epoch', type=int, default=150, help='max training iterations')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate, default=0.0001')
parser.add_argument('--num_workers', type=int, default=2, help='num of workers, default=8')
parser.add_argument('--class_num', type=int, default=6, help='num of classes, landslide=6')
option = parser.parse_known_args()[0]


transform = transforms.Compose([transforms.ToTensor()])

test_path = Path('./Net_test1.json')
test_dataset = SimpleDateset(test_path, transform)
test_dataloader = DataLoader(test_dataset, batch_size=option.batch_size, shuffle=False, num_workers=option.num_workers)


net = ResNet(ResBlock).cuda()
net.load_state_dict(torch.load('AMGC_ResNet18.pth'))
ce_loss = nn.CrossEntropyLoss().cuda()   


net.eval()
test_correct = 0
test_total = 0
correct_01 = 0
loss_test = 0
temp_r = []
temp_p = []
for info in test_dataloader:
    data, label = info
    data = data.type(torch.FloatTensor)
    data, label = Variable(data.cuda()), Variable(label.cuda())
    res = net(data)
    test_loss = ce_loss(res, label)
    _, predicted = torch.max(res.data, 1)
    r = label.tolist()
    p = predicted.tolist()
    temp_r += r
    temp_p += p
    print('real ', r, ' pred', p)
    for r_, p_ in zip(r, p):
        if (r_ <= 2 and p_ <= 2) or (r_ >= 3 and p_ >= 3):
            correct_01 += 1
    test_total += label.size(0)
    test_correct += (predicted == label).sum()
    loss_test += test_loss.item()
        
loss_test /= len(test_dataloader)
test_acc = round(100. * float(test_correct / test_total), 3)
acc_01 = round(100. * correct_01 / test_total, 3)
        
print('Test acc：%.3f%%' % test_acc)
print('0-1 acc: %.3f%%' % acc_01)
print('Test loss: %.5f' % loss_test)