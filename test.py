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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = torch.load('model_aiwaf.pth')
    model = model.to(device)
    model.eval()  # Set model to evaluate mode
    class_names = ['white', 'sqli', 'xss']

    file_name = "data/white64.csv"
    data = pd.read_csv(file_name, nrows = 100)
    data_list = data['data'].values.tolist()
    start_time1 = time.time()
    for str in data_list:
        # str2index  操作，
        input = str2index(str, SEQ_LEN)
        input = np.array(input)
        input = torch.from_numpy(input)
        inputs = input.unsqueeze(0)
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        if preds[0] != 0:
            print(class_names[preds[0]])

        # print(str)
        # print(class_names[preds[0]])
    print("flow:", time.time() - start_time1)

    start_time2 = time.time()
    inputs = []
    for str in data_list:
    # str2index  操作，
        input = str2index(str, SEQ_LEN)
        inputs.append(input)
    inputs = np.array(inputs)
    inputs = torch.from_numpy(inputs)
    inputs = inputs.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    if preds[0] != 0:
        print(class_names[preds[0]])
    print("batch:", time.time() - start_time2)



