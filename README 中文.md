# StatGraph
StatGraph 是一个多视图统计图学习模型，专门用于车载网络入侵的有效和细粒度检测。StatGraph 拥有高性能操作效率，非常适用于车辆环境，能够保护车载 CAN 总线的安全。

更多详情请参见我们的论文 **通过对 CAN 消息进行多视图统计图学习来实现有效的车载入侵检测**。([arXiv:2311.07056](https://arxiv.org/abs/2311.07056))

我们提供了三个文件夹：一个“Dataset”文件夹包含原始数据，“StatGraph-CarHacking”文件夹和“StatGraph-ROAD”文件夹则包含数据处理代码和神经网络相关的代码。

在运行“ModelAdapting/run”文件夹中的代码之前，请在每个“StatGraph-”文件夹里的“dataprocess”文件夹内执行代码，以生成可供神经网络使用的数据。
