# AIWAF_1
`data`文件夹用来存放训练集，以及后续测试数据  
`out_file`文件夹用来存放测试数据的输出结果
`images`文件夹里是几个曲线图
# 运行流程
1. 下载源码  
```bash
git clone https://github.com/CKZfd/AIWAF_1.git
```
2. 安装必须的库
```bash
pip install torch  torchvision  numpy pandas matplotlib
```
3. 从releases 1.0 中下载权重文件放入`data`  
```bash
cd data
wget https://github.com/CKZfd/AIWAF_1/releases/download/1.0/sqli_base64.csv
wget https://github.com/CKZfd/AIWAF_1/releases/download/1.0/white64.csv
wget https://github.com/CKZfd/AIWAF_1/releases/download/1.0/xss_base64.csv
cd -
```
4、训练
```bash
python train.py
```
5、测试
```bash
python data_classification.py
# 请查阅该文件相关注释
```
6、关于训练数据，使用dataloader加载，请参阅文件aiwaf_dataset.py
模型请参阅 model.py, 训练生成的模型文件为model_aiwaf.pth
其余为一些辅助调试和理解的程序


