import numpy as np
import os
import glob
import pandas as pd

def Get_results(path):
    f = open(path,'r',encoding='utf-8')
    text = f.readlines()[-1].strip('\n')
    index = text.find("=")
    result = float(text[index+1:]) #找到最后一行的数值
    return result

arr = np.empty((0,3))
for seed in [42,667,128]:
    for lr in [0.001]:
        file = "cnn1dmodels/data-1_n_epochs-5_aug-150_noise-0.01_supportSize-0_innerSteps-1_lrMeta-{}_lrInner-0.0001_warmUp-0.0_seed-{}".format(lr,seed)
        path = os.path.join(file,"log")
        filename = "data-3-datatest-3_supportSize-0_innerSteps-1_lrInner-0.0001_seed-{}.txt".format(seed)
        path = os.path.join(path,filename)
        print(path)
        result = Get_results(path)

        temp = np.array([lr, seed, result])
        arr = np.vstack((arr, temp))
df = pd.DataFrame(arr)
df.to_csv("FD001-TRANSFER-FD003_BaseModel.csv",
          header=["lrMeta", "seed", "result"], index=False)
