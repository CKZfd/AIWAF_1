from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
start_time = time.time()
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score,f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
# from scipy import interp
from numpy import interp

from aiwaf_dataset import aiwaf_class
import torch
from torch.utils.data.dataset import random_split
import numpy as np
import matplotlib.pyplot as plt

"""https://blog.csdn.net/qq_38410428/article/details/88106395"""




if __name__ == "__main__":

    SEQ_LEN = 100
    nb_classes = 3

    aiwaf_datasets = aiwaf_class(seq_len=SEQ_LEN)
    train_len = int(len(aiwaf_datasets) * 0.9)
    aiwaf_datasets_train, aiwaf_datasets_val = \
        random_split(aiwaf_datasets, [train_len, len(aiwaf_datasets) - train_len])
    dataset_sizes = {'train': len(aiwaf_datasets_train), 'val': len(aiwaf_datasets_val)}
    batch_size = dataset_sizes['val']
    dataloaders = {'train': torch.utils.data.DataLoader(aiwaf_datasets_train, batch_size=batch_size,
                                                        shuffle=False, num_workers=2),
                   'val': torch.utils.data.DataLoader(aiwaf_datasets_val, batch_size=batch_size,
                                                      shuffle=False, num_workers=2)}

    class_names = ['white', 'sqli', 'xss']

    X_valid, Y_valid = next(iter(dataloaders['val']))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('model_aiwaf.pth')
    model = model.to(device)
    model.eval()  # Set model to evaluate mode

    X_valid = X_valid.to(device)
    outputs = model(X_valid)
    outputs = outputs.cpu()
    _, Y_pred = torch.max(outputs, 1)

    # Binarize the output
    Y_valid = label_binarize(Y_valid, classes=[i for i in range(nb_classes)])
    Y_pred = label_binarize(Y_pred, classes=[i for i in range(nb_classes)])

    # micro：多分类　　
    # weighted：不均衡数量的类来说，计算二分类metrics的平均
    # macro：计算二分类metrics的均值，为每个类给出相同权重的分值。
    precision = precision_score(Y_valid, Y_pred, average='micro')
    recall = recall_score(Y_valid, Y_pred, average='micro')
    f1_score = f1_score(Y_valid, Y_pred, average='micro')
    accuracy_score = accuracy_score(Y_valid, Y_pred)
    print("Precision_score:", precision)
    print("Recall_score:", recall)
    print("F1_score:", f1_score)
    print("Accuracy_score:", accuracy_score)

    # roc_curve:真正率（True Positive Rate , TPR）或灵敏度（sensitivity）
    # 横坐标：假正率（False Positive Rate , FPR）
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_valid[:, i], Y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_valid.ravel(), Y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nb_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= nb_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.6f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.6f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(nb_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.6f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig("images/ROC/ROC_3分类.png")
    plt.show()
    print("--- %s seconds ---" % (time.time() - start_time))


