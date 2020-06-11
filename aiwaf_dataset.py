import pandas as pd
import numpy as np
import base64
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


# def get_longest_element(item_list): #定义获取列表中最长元素的函数
#     len_list=map(len,item_list) #计算list每个元素的长度
#     li=list(len_list) #实例化\
#     li.sort()
#     print(len(li),li)
#     return item_list[np.argmax(li)], len(item_list[np.argmax(li)]) #返回最长元素



# white64_data = pd.read_csv("data/white64.csv")   # 688312  [32 - 1272] 均匀
# sqli_data = pd.read_csv("data/sqli_base64.csv")  # 23150   [8 - 544]   8-380较均匀
# xss_data = pd.read_csv("data/xss_base64.csv")    # 126151  [8 - 8000]  8-1000 较均匀
# # 可以考虑长度为500的WordVec,太长的砍掉，不足的补PAD
#
# white64_list = white64_data['data'].values.tolist()
# sqli_list = sqli_data['data'].values.tolist()
# xss_list = xss_data['data'].values.tolist()


word2index = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
              'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16,
              'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24,
              'Y': 25, 'Z': 26, 'a': 27, 'b': 28, 'c': 29, 'd': 30, 'e': 31, 'f': 32,
              'g': 33, 'h': 34, 'i': 35, 'j': 36, 'k': 37, 'l': 38, 'm': 39, 'n': 40,
              'o': 41, 'p': 42, 'q': 43, 'r': 44, 's': 45, 't': 46, 'u': 47, 'v': 48,
              'w': 49, 'x': 50, 'y': 51, 'z': 52, '0': 53, '1': 54, '2': 55, '3': 56,
              '4': 57, '5': 58, '6': 59, '7': 60, '8': 61, '9': 62, '+': 63, '/': 64,
              '=': 65, 'PAD': 66}
index2word = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H',
              9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P',
              17: 'Q', 18: 'R', 19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X',
              25: 'Y', 26: 'Z', 27: 'a', 28: 'b', 29: 'c', 30: 'd', 31: 'e', 32: 'f',
              33: 'g', 34: 'h', 35: 'i', 36: 'j', 37: 'k', 38: 'l', 39: 'm', 40: 'n',
              41: 'o', 42: 'p', 43: 'q', 44: 'r', 45: 's', 46: 't', 47: 'u', 48: 'v',
              49: 'w', 50: 'x', 51: 'y', 52: 'z', 53: '0', 54: '1', 55: '2', 56: '3',
              57: '4', 58: '5', 59: '6', 60: '7', 61: '8', 62: '9', 63: '+', 64: '/',
              65: '=', 66: 'PAD'}


def str2index(data, seq_len):
    res = []
    n = len(data)
    i = 0
    while i < seq_len:
        if i < n:
            res.append(word2index[data[i]])
        else:
            res.append(word2index['PAD'])
        i += 1
    return res

def make_dataset():
    white64_data = pd.read_csv("data/white64.csv")  # 688312  [32 - 1272] 均匀
    sqli_data = pd.read_csv("data/sqli_base64.csv")  # 23150   [8 - 544]   8-380较均匀
    xss_data = pd.read_csv("data/xss_base64.csv")  # 126151  [8 - 8000]  8-1000 较均匀
    # 可以考虑长度为500的WordVec,太长的砍掉，不足的补PAD

    white64_list = white64_data['data'].values.tolist()
    sqli_list = sqli_data['data'].values.tolist()
    xss_list = xss_data['data'].values.tolist()
    return white64_list, sqli_list, xss_list

class aiwaf_class(data.Dataset):
    def __init__(self, seq_len=500, str2index = str2index, seed=None):
        self.str2index = str2index
        self.seq_len = seq_len
        self.white64_list, self.sqli_list, self.xss_list = make_dataset()
        self.white64_len = len(self.white64_list)
        self.sqli_len = len(self.sqli_list)
        self.xss_len = len(self.xss_list)
        if seed is not None:
            np.random.seed(seed)

    def __getitem__(self, _):  # index随机生成
        index_sub = np.random.randint(0, 3)  # 3分类， white, sqli, xss
        label = index_sub
        if index_sub == 0:
            index = np.random.randint(0, self.white64_len)  # white 训练集长度
            data = self.white64_list[index]
        if index_sub == 1:
            index = np.random.randint(0, self.sqli_len)  # sqli 训练集长度
            data = self.sqli_list[index]
        if index_sub == 2:
            index = np.random.randint(0, self.xss_len)  # xss 训练集长度
            data = self.xss_list[index]

        # str2index  操作，
        data = self.str2index(data, self.seq_len)
        data = np.array(data)
        data = torch.from_numpy(data)
        # data = data.float()
        return data, label

    def __len__(self):
        return self.white64_len + self.sqli_len + self.xss_len






