"""
需要注意字段
input_file = "data/payload2.csv"
output_file = "out_file/payload2_withlabel.csv"
data_list = data['payload'].values.tolist()
id_list = data['id'].values.tolist()
data_dict = {"id": id_list, "payload": data_list, "label": output_labels}
df = pd.DataFrame(data_dict, columns=['id', 'payload', 'label'])
"""
import torch
import torch.nn
import pandas as pd
import numpy as np
import time

word2index = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
              'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16,
              'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24,
              'Y': 25, 'Z': 26, 'a': 27, 'b': 28, 'c': 29, 'd': 30, 'e': 31, 'f': 32,
              'g': 33, 'h': 34, 'i': 35, 'j': 36, 'k': 37, 'l': 38, 'm': 39, 'n': 40,
              'o': 41, 'p': 42, 'q': 43, 'r': 44, 's': 45, 't': 46, 'u': 47, 'v': 48,
              'w': 49, 'x': 50, 'y': 51, 'z': 52, '0': 53, '1': 54, '2': 55, '3': 56,
              '4': 57, '5': 58, '6': 59, '7': 60, '8': 61, '9': 62, '+': 63, '/': 64,
              '=': 65, 'PAD': 66}

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


if __name__ == '__main__':
    SEQ_LEN = 100
    batch_size = 1000
    # 需要测试的文件，以及测试结果输出文件
    input_file = "data/payload2.csv"
    output_file = "out_file/payload2_withlabel.csv"


    # 模型加载
    device = torch.device("cpu")
    model = torch.load('model_aiwaf.pth')
    model = model.to(device)
    model.eval()  # Set model to evaluate mode
    class_names = ['white', 'sqli', 'xss']

    start_time = time.time()
    # 数据读入，data_list里data['payload'] 之'payload' 和['id'] 需要根据表格头标签进行修改
    data = pd.read_csv(input_file)
    data_list = data['payload'].values.tolist()
    id_list = data['id'].values.tolist()
    output_labels = []

    charsIndexes = []
    for i, chars in enumerate(data_list):
        charsIndex = str2index(chars, SEQ_LEN)
        charsIndexes.append(charsIndex)
        if (i+1) % batch_size == 0:
            inputs = np.array(charsIndexes)
            inputs = torch.from_numpy(inputs)
#             print(i, inputs, len(inputs))
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            output_labels += preds.numpy().tolist()
            charsIndexes = []
    if (i + 1) % batch_size != 0:
        inputs = np.array(charsIndexes)
        inputs = torch.from_numpy(inputs)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        output_labels += preds.numpy().tolist()

    print(output_labels.count(0))
    print(output_labels.count(1))
    print(output_labels.count(2))

#     for i in range(900, len(data_list)):
#         output_labels.append(" ")

    # 表格头行，每一列分别是
    data_dict = {"id": id_list, "payload": data_list, "label": output_labels}
    df = pd.DataFrame(data_dict, columns=['id', 'payload', 'label'])
    df.to_csv(output_file, index=False)
    print(time.time() - start_time)




