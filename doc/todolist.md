#### 训练过程
- Batch size ： 越小越不会过拟合
- learning rate
- dropout probability
- loss function : `nn.BCELossWithLogits()`

#### user friendly
- F1 score 和 accuracy 
- 对于每一个类别的F1 score 单独展示
- 中间结果的保存、可视化
- 一个console的ui
- 根据标签的权重展示f1 score, `average='weighted'`
- 阈值调优了之后的结果

#### 改进
- 下游resnet
- 数据集的扩充
- 阈值调优，每个标签根据训练数据的分布，设置不同的阈值

#### 不同情感分类的对比
- Emotion : 28 classes
- Sentiment : `ambiguous` `negative` `neutral` `positive` `not mentioned`
- Ekman Emotion : `anger` `disgust` `fear` `joy` `sadness` `surprise`