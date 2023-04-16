import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
import math
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
plt.rcParams['axes.unicode_minus'] = False

# base model
data = np.loadtxt("/data/moy20/moy20/Meta-Learning/Meta-prognosis-main/transformer_models/data-1_n_epochs-5_aug-150_noise-0.01_supportSize-0_innerSteps-1_lrMeta-0.001_lrInner-0.0001_warmUp-0.0_seed-667/result_TEST_1.txt",skiprows=1)
res = math.sqrt(sum(data[:,0]**2)/100)
print(res)

# base model + finetune
data = np.loadtxt("/data/moy20/moy20/Meta-Learning/Meta-prognosis-main/transformer_models/data-1_n_epochs-5_aug-150_noise-0.01_supportSize-2_innerSteps-1_lrMeta-0.001_lrInner-0.0001_warmUp-0.0_seed-667/result_TEST_1.txt",skiprows=1)


# FD001
def fd001():
    path = '/data/moy20/moy20/Informer/RULPredict/Transformer/dataset-1/epochmax-15-batch-8-dm-128-df-512-dp-0.2-l-1-heads-4-lr-0.001-noise-0.1-seed-667/'
    pre = np.loadtxt(path+'predict.txt')
    true = np.loadtxt(path+"true.txt")
    return pre,true
def fd003():
    path = '/data/moy20/moy20/Informer/RULPredict/Transformer/dataset-3/epochmax-15-batch-8-dm-64-df-512-dp-0.2-l-1-heads-4-lr-0.001-noise-0.1-seed-42/'
    pre = np.loadtxt(path+'predict.txt')
    true = np.loadtxt(path+"true.txt")
    return pre,true

def fd002():
    path = '/data/moy20/moy20/Informer/RULPredict/Transformer/dataset-2/epochmax-10-batch-16-dm-64-df-256-dp-0.1-l-2-heads-2-lr-0.002-noise-0.0-seed-217/'
    pre = np.loadtxt(path+'predict.txt')
    true = np.loadtxt(path+"true.txt")
    return pre,true

def fd004():
    path = '/data/moy20/moy20/Informer/RULPredict/Transformer/dataset-4/epochmax-10-batch-16-dm-64-df-512-dp-0.2-l-1-heads-2-lr-0.001-noise-0.0-seed-217/'
    pre = np.loadtxt(path+'predict.txt')
    true = np.loadtxt(path+"true.txt")
    return pre,true

def dealwith(pre,true):
    index = true.argsort()
    pre = pre[index]
    true.sort()
    # 真实值大于125的都设置为125
    true = np.where(true>125,125,true)
    return pre,true

def drawcurve():

    plt.figure(constrained_layout=True)
    font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 8,
         }

    plt.subplot(221)
    pre,true = fd001()
    pre,true = dealwith(pre,true)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xlim(0,100)
    plt.grid(linestyle="-.")
    plt.plot(np.arange(1,len(pre)+1,1),true,"b")
    plt.scatter(np.arange(1,len(pre)+1,1),pre,s=10,c="r",marker="x")
    plt.legend(labels=["True RUL(Piece-Wise)","Predict RUL"],loc="lower right",fontsize=8)
    plt.title("FD001",font2)

    plt.subplot(222)
    pre,true = fd002()
    pre,true = dealwith(pre,true)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xlim(0,260)
    plt.grid(linestyle="-.")
    plt.plot(np.arange(1,len(pre)+1,1),true,"b")
    plt.scatter(np.arange(1,len(pre)+1,1),pre,s=10,c="r",marker="x")
    plt.legend(labels=["True RUL(Piece-Wise)","Predict RUL"],loc="lower right",fontsize=8)
    plt.title("FD002",font2)

    plt.subplot(223)
    pre,true = fd003()
    pre,true = dealwith(pre,true)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xlim(0,100)
    plt.grid(linestyle="-.")
    plt.plot(np.arange(1,len(pre)+1,1),true,"b")
    plt.scatter(np.arange(1,len(pre)+1,1),pre,s=10,c="r",marker="x")
    plt.legend(labels=["True RUL(Piece-Wise)","Predict RUL"],loc="lower right",fontsize=8)
    plt.title("FD003",font2)

    plt.subplot(224)
    pre,true = fd004()
    pre,true = dealwith(pre,true)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xlim(0,250)
    plt.grid(linestyle="-.")
    plt.plot(np.arange(1,len(pre)+1,1),true,"b")
    plt.scatter(np.arange(1,len(pre)+1,1),pre,s=10,c="r",marker="x")
    plt.legend(labels=["True RUL(Piece-Wise)","Predict RUL"],loc="lower right",fontsize=8)
    plt.title("FD004",font2)

    plt.savefig("figs/FullAttention.jpg")

# pre,true = fd001()
# drawcurve(pre,true,"fd001")
# pre,true = fd003()
# drawcurve(pre,true,"fd003")
# pre,true = fd002()
# drawcurve(pre,true,"fd002")
# pre,true = fd004()
# drawcurve(pre,true,"fd004")

