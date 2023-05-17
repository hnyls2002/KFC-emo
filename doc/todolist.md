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

#### 改进
- 下游任务一层线性层or其他的设计
- 数据集的扩充
- 对于标签个数较小的时候的尝试优化

#### 不同情感分类的对比
- Emotion : 28 classes
- Sentiment : `ambiguous` `negative` `neutral` `positive` `not mentioned`
- Ekman Emotion : `anger` `disgust` `fear` `joy` `sadness` `surprise`