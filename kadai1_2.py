from pickletools import optimize
import numpy as np
from mnist import load_mnist
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# Adamの実装
class Adam:

    # インスタンス変数を定義
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.lr = lr # 学習率
        self.beta1 = beta1 # mの減衰率
        self.beta2 = beta2 # vの減衰率
        self.iter = 0 # 試行回数を初期化
        self.m = None # モーメンタム
        self.v = None # 適合的な学習係数
    
    # パラメータの更新メソッドを定義
    def update(self, params, grads):
        # mとvを初期化
        if self.m is None: # 初回のみ
            self.m = {}
            self.v = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val) # 全ての要素が0
                self.v[key] = np.zeros_like(val) # 全ての要素が0
        
        # パラメータごとに値を更新
        self.iter += 1 # 更新回数をカウント
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter) # 式(6)の学習率の項
        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key] # 式(1)
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2) # 式(2)
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7) # 式(6)

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def softmax(x):
    if x.ndim == 2:
      x = x.T
      x = x - np.max(x, axis=0)
      y = np.exp(x) / np.sum(np.exp(x), axis=0)
      return y.T
    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


#予測を行う関数
def predict(x,params):
        W1, W2 = params['W1'], params['W2']
        b1, b2 = params['b1'], params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

#損失関数の計算
# x:入力データ, t:教師データ
def loss(x,t,params):
    y = predict(x,params)
        
    return cross_entropy_error(y, t)

#予測の精度を求める関数
def accuracy(x,t,params):
        y = predict(x,params)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

#勾配を求める関数
def gradient(x,t,params):
        W1, W2 = params['W1'], params['W2']
        b1, b2 = params['b1'], params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        batch_num = x.shape[0]

        # backward
        dy = (y - t)/batch_num #データ一個当たりの誤差を求める
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads




#重みとバイアスの初期化
input_size=2 #入力層
hidden_size=20  #中間層
output_size=2 #出力層

# 重みの初期値の標準偏差を指定
weight_init_std=0.1

params = {}
params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
params['b1'] = weight_init_std * np.random.randn(hidden_size)
params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
params['b2'] = weight_init_std * np.random.randn(output_size)

#データ分割
data = pd.read_table("2class.txt", sep=" ", header=None)
x = data[0].values  
y = data[1].values
t = data[2].values


x_train, x_test,y_train,y_test,t_train, t_test = train_test_split(x,y,t,train_size=0.9) 

#print(x_train.shape)#1801

train_x=np.array([])
for j in range(len(x_train)):
    train_x = np.append(train_x,[ x_train[j] , y_train[j] ] )

train_x=train_x.reshape(len(x_train),2).astype(float)

test_x=np.array([])
for j in range(len(x_test)):
    test_x = np.append(test_x,[ x_test[j] , y_test[j] ] )

test_x=test_x.reshape(len(x_test),2).astype(float)

train_t=np.array([])

for j in range(len(t_train)):
    train_t = np.append(train_t,[ 1-t_train[j] , t_train[j] ] )

train_t=train_t.reshape(len(t_train),2).astype(float)

test_t=np.array([])

for j in range(len(t_test)):
    test_t = np.append(test_t,[ 1-t_test[j] , t_test[j] ] )

test_t=test_t.reshape(len(t_test),2).astype(float)

num=90000
learning_rate=0.1
batch_size=100
train_size=train_x.shape[0]
epoch=int(max(train_size / batch_size, 1))#18
epoch_num=1

train_loss_list = []
test_loss_list = []
train = []
test = []
epoch_num_list=[]

optimizer=Adam()
# 開始
start_time = time.perf_counter()
 
# ダミー処理
time.sleep(1)
 
for i in range(num):
    #バッチ処理
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = train_x[batch_mask]
    t_batch = train_t[batch_mask]
    
    #勾配を求める
    grads=gradient(x_batch,t_batch,params)

    # 更新
    """"
    for key in ('W1', 'b1', 'W2', 'b2'):
       params[key] -= learning_rate * grads[key]
    """
    optimizer.update(params,grads)
    #精度を表示
    if i % epoch == 0:
        train_acc = accuracy(train_x,train_t,params)
        test_acc = accuracy(test_x, test_t,params)
        loss_train=loss(x_batch, t_batch,params)
        train.append(train_acc)
        test.append(test_acc)
        train_loss_list.append(loss(train_x,train_t,params))
        test_loss_list.append(loss(test_x,test_t,params))
        print(epoch_num, train_acc, test_acc,loss(train_x,train_t,params))
        epoch_num_list.append(epoch_num)
        epoch_num=epoch_num+1
# 修了
end_time = time.perf_counter()
 
# 経過時間を出力(秒)
elapsed_time = end_time - start_time
print(elapsed_time)

markers = ['o', 'x', '^']


for i in range(len(x)):
    plt.scatter(x[i], y[i], marker=markers[t[i]], 
                s=50, label='class ' + str(t[i])) # データセット
"""""
plt.ylim([0,1])
#plt.ylim([0,1.5])
#plt.plot(epoch_num_list,train,color='b',label='train')
#plt.plot(epoch_num_list,test ,color='r',label='test')
plt.plot(epoch_num_list,train_loss_list,color='b',label='train')
plt.plot(epoch_num_list,test_loss_list ,color='r',label='test')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()
"""
