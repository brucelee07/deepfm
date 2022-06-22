## TIPS

- 程序开发采用python, pytorch

- models.py 模型定义代码文件

- dataset.py 数据dataset代码文件

- main.py 主程序，训练model

## 简介

### models.py

- Fm   `fm 部分参考文献和torchrec` 
- Dense   `线性层构建的deep部分`
- Deepfm   `embedding层 -> feature embedding >(dense, fm) -> out`


### main.py

- 定义sparse feats 字典，用来定义embedding层的维度 

- 目前文件中dense feats 为空， userID数据庞大，引入embedding层，可能使参数非常大，不方便训练

- 定义 hidden units 为 Dense 参数

- 定义auc, logloss函数

- 考虑数据量太大，采用chunsize迭代dataframe处理，但是要考虑embedding层的维度

- 将数据拆分train, val, 训练模型
