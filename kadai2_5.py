from __future__ import print_function
from statistics import mode
from tkinter import Variable
#from math import gamma #コマンドライン引数を受け取る処理をするモジュール
import torch # ライブラリ「PyTorch」のtorchパッケージをインポート
import torch.nn as nn  # 「ニューラルネットワーク」モジュールの別名定義
import torch.nn.functional as F #様々な関数を持つクラス
import torch.optim as optim#adam,SGDなどの最適化手法をもつモジュール
from torchvision import datasets,transforms#データの前処理に必要なモジュール
from torch.optim.lr_scheduler import StepLR#学習率の更新を行う関数
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time

class Net(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Net, self).__init__()#親のクラスを継承
        self.fc1 = nn.Linear(input_size,hidden_size) # Linearは「全結合層」を指す(入力層、出力層)
        self.fc2 = nn.Linear(hidden_size,output_size)# Linearは「全結合層」を指す(入力層、出力層)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        output = self.fc2(x)
        #output =torch.log_softmax(x, dim=1)
        return output

    #予測の精度を求める関数
    
    def accuracy(self,x,t):
        accuracy=0
        y = self.forward(x)
        #y=y.cpu().data.numpy()
        y =torch.argmax(y, dim=1)#dim=1=列方向に見ていく（横）
        #print(list(y))
        #t=t.cpu().data.numpy()
        for i in range(len(y)):
            if(y[i]==t[i]):
                accuracy+=1
        return 1

#データ分割
train_images = np.load('kmnist-train-imgs.npz')['arr_0']
train_labels = np.load('kmnist-train-labels.npz')['arr_0']
test_images = np.load('kmnist-test-imgs.npz')['arr_0']
test_labels = np.load('kmnist-test-labels.npz')['arr_0']

transform=transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))
        ])

#train_images = torch.tensor(train_images)#Tensor型に変換,微分可能にする
#train_images=train_images.to(torch.float)
#test_images = torch.tensor(test_images)#Tensor型に変換,微分可能にする
#test_images=test_images.to(torch.float)
train_labels = torch.LongTensor(train_labels)#Tensor型に変換
test_labels = torch.LongTensor(test_labels)#Tensor型に変換

dataset1=transform(train_images)
dataset2=transform(test_images)
train_set = torch.utils.data.TensorDataset(dataset1,train_labels)
test_set = torch.utils.data.TensorDataset(dataset2,test_labels)

batch_size=100
train_size=dataset1.shape[0]#2702
epoch=int(max(int(train_size / batch_size), 1))#27
lr=0.1
seed=0
epoch_num=1

train_loader = torch.utils.data.DataLoader(train_set,batch_size, shuffle = True,drop_last=True)#drop_last;余ったデータを捨てる
test_loader = torch.utils.data.DataLoader(test_set,batch_size, shuffle = True,drop_last=True)
use_cuda =torch.cuda.is_available()#cudaを使えと指定されcudaが使える場合にcudaを使用
torch.manual_seed(seed)#疑似乱数を作る時に使う、元となる数字。 シード値が同じなら常に同じ乱数が作られる。(再現性がある)

device = torch.device("cuda" if use_cuda else "cpu")#GPUを指定なければCPU
#device=torch.device("cpu")
#GPUに送る

train_x=dataset1.to(device)
train_t=train_labels.to(device)#修正
test_x=dataset2.to(device)
test_t=test_labels.to(device)#修正

train_loss_list = []
test_loss_list = []
train = []
test = []
epoch_num_list=[]

model = Net(2,20,3).to(device)#netインスタンス生成。modelはレイヤーの構成親クラスを継承
#推奨params lr=0.01β1=0.9,β2=0.999ϵ=1e−8
optimizer = optim.Adam(model.parameters(),lr=0.01)#最適化手法,model.parameters():自動で重みとバイアスを設定してくれる
#scheduler = StepLR(optimizer, step_size=1,gamma=0.1)#step_size：更新タイミングのエポック数,gamma：更新率
criterion = nn.CrossEntropyLoss()
#print(list(model.parameters()))
# 開始
start_time = time.perf_counter()
 
# ダミー処理
time.sleep(1)
#output = model.forward(train_x)
#print(list(output))
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []
total_epoch=int(1e+4)

for epoch_num in range(total_epoch+1):
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0

     # 損失和
    train_epoch_loss = 0.0
        # 正解数
    train_epoch_corrects = 0

    model.train()
    for k, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = F.softmax(model.forward(data),dim=1)
        loss =criterion(model.forward(data),labels)
        loss.backward()
        optimizer.step()
        _,preds = torch.max(outputs, 1)#torch.maxは最大値（テンソル）とその要素位置の２つを返します_で最大値を受け取っている
        train_loss += loss.item() * data.size(0)
        # 正解数の合計を更新
        train_acc += torch.sum(preds == labels.data)

        # epochごとのlossと正解率を表示
    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_acc = train_acc.double() / len(train_loader.dataset)
    
    
    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs =F.softmax(model.forward(data),dim=1)
            loss =criterion(model.forward(data),labels)
            _,preds = torch.max(outputs, 1)
            val_loss += loss.item() * data.size(0)
            val_acc += torch.sum(preds == labels.data)
    avg_val_loss = val_loss / len(test_loader.dataset)
    avg_val_acc = val_acc.double() / len(test_loader.dataset)

    print('{} train_Loss: {:.6f} train_Acc: {:.6f} test_Loss: {:.6f} test_Acc: {:.6f}'.format(epoch_num,avg_train_loss,avg_train_acc.item(),avg_val_loss,avg_val_acc.item()))
    epoch_num_list.append(epoch_num)
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc.item())
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(avg_val_acc.item())
    epoch_num=epoch_num+1
# 修了
end_time = time.perf_counter()
 
# 経過時間を出力(秒)
elapsed_time = end_time - start_time
print('elapsed time {}'.format(elapsed_time))
#print(list(model.parameters()))

train_acc_list=torch.tensor(train_acc_list)#tensor型に変更
val_acc_list=torch.tensor(val_acc_list)#tensor型に変更
train_loss_list=torch.tensor(train_loss_list)#tensor型に変更
val_loss_list=torch.tensor(val_loss_list)#tensor型に変更

train_acc_list =train_acc_list.cpu().data.numpy()#cpuに転送
val_acc_list = val_acc_list.cpu().data.numpy()#cpuに転送
train_loss_list = train_loss_list.cpu().data.numpy()#cpuに転送
val_loss_list = val_loss_list.cpu().data.numpy()#cpuに転送

plt.ylim([0,1])
plt.plot(epoch_num_list,train_acc_list,color='b',label='train')
plt.plot(epoch_num_list,val_acc_list ,color='r',label='test')
#plt.plot(epoch_num_list,train_loss_list,color='b',label='train')
#plt.plot(epoch_num_list,val_loss_list ,color='r',label='test')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

