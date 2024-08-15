You can generate the node vector files required for training through "generate attack nodes.py" and "generate attack nodes.py", which process abnormal and normal data sets respectively and include three parts: graph property generation, data slice and data movement.

And then, you can get the node vector files after batch data merging through our "merge node vectors.py". Its output is three csv files named "train_nodes", "val_nodes" and "test_nodes".

generate normal nodes 50_40: 此代码包括数据清洗、图形结构创建、属性提取，以及将数据分割为训练集和验证集等步骤。
merge node vectors 50_40: 此代码段主要用于将多个小的CSV文件合并成大型的训练、验证和测试数据集。它通过读取指定路径下的所有文件，将它们合并为一个Pandas DataFrame，并将结果保存为一个新的CSV文件。