import matplotlib.pyplot as plt
import json
import numpy as np

file = open('train_acc.txt', 'r')
train_acc = file.read()
train_acc = json.loads(train_acc)
file.close()
file = open('train_loss.txt', 'r')
train_loss = file.read()
train_loss = json.loads(train_loss)
file.close()
file = open('val_acc.txt', 'r')
val_acc = file.read()
val_acc = json.loads(val_acc)
file.close()

train_loss = [loss/51200 for loss in train_loss]
train_acc = [acc/512 for acc in train_acc]
val_acc = [acc/512 for acc in val_acc]

x = [i*100 for i in range(1,len(train_acc)+1)]
x2 = [i for i in range(len(val_acc))]

plt.figure()
plt.plot(x, train_loss, 'm--')
plt.xlim((0, 7000))
plt.ylim((0, 1))
plt.title('Train Loss')
plt.xlabel('train_step')
plt.ylabel('train_loss')
plt.savefig("images/TrainLoss.png")
plt.show()

plt.figure()
plt.plot(x, train_acc, 'm--')
plt.xlim((0, 7000))
plt.ylim((50, 100))
plt.title('Train Accuracy')
plt.xlabel('train_step')
plt.ylabel('train_accuracy %')
plt.savefig("images/TrainAccuracy.png")
plt.show()

plt.figure()
plt.plot(x2, val_acc, 'm--')
plt.xlim((0, 4))
plt.ylim((50, 100))
plt.title('Valid Accuracy')
plt.xlabel('valid_epoch')
plt.ylabel('valid_accuracy %')
plt.savefig("images/ValidAccuracy.png")
plt.show()

