# LiPar

LiPar是一个轻量级的并行学习模型，用于实际车载网络入侵检测。LiPar具有出色的检测性能、运行效率和轻量级模型大小，可以实际适应车载环境，并保护车载CAN总线安全。

您可以在我们的论文**LiPar：一个轻量级并行学习模型，用于实际车载网络入侵检测**中查看详细信息。([arXiv:2311.08000](https://arxiv.org/abs/2311.08000))

## 数据集

我们使用的数据集是[汽车黑客数据集](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset)（您可以通过链接查看此数据集的详细信息）。我们还在`./OriginalDataset/`上传了数据文件。由于上传文件大小的限制，我们将每个数据文件压缩成`.rar`文件。您可以通过解压文件获取原始数据。

## 数据处理

数据处理代码上传在`./DataProcessing/`。以下是我们的数据处理步骤：

1. 数据预处理和数据清洗：`data_process.py`用于处理攻击数据集，包括`DoS_Attack_dataset`、`Fuzzy_Attack_dataset`、`Spoofing_the_RPM_gauge_dataset`和`Spoofing_the_drive_gear_dataset`。`data_process_normal.py`仅用于处理`normal`数据集。您可以在代码中的`df.to_csv`函数更改`file_path`和新文件名，逐个预处理数据集。然后，它将分别生成五个预处理数据文件。生成的数据集可以在压缩文件`./DataProcessing/PreprocessedData.rar`中找到。
2. 图像数据生成：`img_generator_seq.py`用于将一维数据顺序处理成RGB图像数据。对于攻击数据集，既有攻击消息也有正常消息。因此，如果一个图像完全由正常消息组成，我们将其标记为正常图像。否则，我们将其标记为攻击图像。因此，每个攻击数据集可以生成两组图像。您可以在`image_path_attack`和`image_path_normal`中设置不同的目录地址来存储生成的正常和攻击图像。每组新生成的图像将按序列从1到`n`（`n`是该组中图像的总数）命名。处理一个数据文件后，您可以在`file_path`中更改文件名和路径来处理其他数据文件。我们使用的文件是上一步获得的预处理数据文件。对于正常数据集，程序当然只会生成一组正常图像。最后，您将在不同的目录中获得9组图像。
3. 数据集分割：我们用于存储所有图像的目录是`./data_sequential_img/train/`。然后，我们需要从所有图像数据中划分训练集、验证集和测试集。`split_trainset.py`用于将总图像数据的30%划分到名为`./data_sequential_img/val/`的验证和测试集目录中。此外，`split_testset.py`用于将`val`集中的$\frac{1}{3}$图像划分到名为`./data_sequential_img/test/`的测试集中。最终，训练集、验证集和测试集的图像比例为`7:2:1`。您还可以通过修改`Train_Dir`、`Val_Dir`和`Test_Dir`来更改路径和目录名。
