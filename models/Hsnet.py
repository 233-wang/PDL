import torch
import torch.nn as nn
import torch.nn.functional as F

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu'):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.activation = activation

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         if self.activation == 'relu':
#             return F.relu(x)
#         elif self.activation == 'mish':
#             return x * torch.tanh(F.softplus(x))  # Mish activation function
#         elif self.activation == 'hard_swish':
#             return x * F.relu6(x + 3) / 6  # Hard Swish activation function
#         else:
#             return x

# class HierarchicalSplitBlock(nn.Module):
#     def __init__(self, channels):
#         super(HierarchicalSplitBlock, self).__init__()
#         self.split_channels = channels // 2  # Assuming an equal split for simplicity
        
#         # Doubling the paths or increasing the channel output
#         self.path1 = ConvBlock(self.split_channels, self.split_channels * 2, 3, padding=1, activation='mish')
#         self.path2 = ConvBlock(self.split_channels, self.split_channels * 2, 3, padding=1, activation='mish')

#     def forward(self, x):
#         # Split feature maps
#         x1, x2 = x.chunk(2, dim=1)
        
#         # Process through different paths
#         x1 = self.path1(x1)
#         x2 = self.path2(x2)
        
#         # Concatenate results
#         out = torch.cat([x1, x2], dim=1)
#         return out

# class Hsnet(nn.Module):
#     def __init__(self, num_classes):
#         super(Hsnet, self).__init__()
#         self.initial_conv = ConvBlock(3, 64, 1)  # Adjust according to input channel size
#         self.hierarchical_block = HierarchicalSplitBlock(64)
#         self.final_conv = ConvBlock(128, 128, 1)  # Adjusted to maintain channel depth
#         self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
#         self.classifier = nn.Linear(128, num_classes)  # Classifier layer

#     def forward(self, x):
#         x = self.initial_conv(x)
#         x = self.hierarchical_block(x)
#         x = self.final_conv(x)
#         x = self.global_avg_pool(x)
#         x = x.view(x.size(0), -1)  # Flatten the output for the classifier
#         x = self.classifier(x)
#         return x


# V0.2
# SE注意力机制模块
# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super(SEBlock, self).__init__()
#         self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
#         self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)

#     def forward(self, x):
#         out = x.mean(dim=(2, 3), keepdim=True)
#         out = F.relu(self.fc1(out))
#         out = torch.sigmoid(self.fc2(out))
#         return x * out

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu'):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.activation = activation

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         if self.activation == 'relu':
#             return F.relu(x)
#         elif self.activation == 'mish':
#             return x * torch.tanh(F.softplus(x))  # Mish activation function
#         elif self.activation == 'hard_swish':
#             return x * F.relu6(x + 3) / 6  # Hard Swish activation function
#         else:
#             return x

# class HierarchicalSplitBlock(nn.Module):
#     def __init__(self, channels):
#         super(HierarchicalSplitBlock, self).__init__()
#         self.split_channels = channels // 2
#         self.path1 = ConvBlock(self.split_channels, self.split_channels, 1, activation='relu')
#         self.path2 = ConvBlock(self.split_channels, self.split_channels, 1, activation='relu')

#     def forward(self, x):
#         x1, x2 = x.chunk(2, dim=1)
#         x1 = self.path1(x1)
#         x2 = self.path2(x2)
#         out = torch.cat([x1, x2], dim=1)
#         return out

# class BalancedLayer(nn.Module):
#     def __init__(self, channels):
#         super(BalancedLayer, self).__init__()
#         self.fc = nn.Conv2d(channels, channels, kernel_size=1)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         weights = self.softmax(self.fc(x.mean(dim=(2, 3), keepdim=True)))
#         return x * weights

# class Hsnet(nn.Module):
#     def __init__(self, num_classes):
#         super(Hsnet, self).__init__()
#         self.initial_conv = ConvBlock(3, 32, 1)  # Reduced initial channels
#         self.hierarchical_block = HierarchicalSplitBlock(32)
#         self.se_block = SEBlock(32)
#         self.balanced_layer = BalancedLayer(32)
#         self.final_conv = ConvBlock(32, 64, 1)  # Adjusted final channels
#         self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.classifier = nn.Linear(64, num_classes)

#     def forward(self, x, is_feat=False, preact=False):
#         feats = []
#         x = self.initial_conv(x)
#         if is_feat:
#             feats.append(x)
        
#         x = self.hierarchical_block(x)
#         if is_feat:
#             feats.append(x)
        
#         x = self.se_block(x)
#         if is_feat:
#             feats.append(x)
        
#         x = self.balanced_layer(x)
#         if is_feat:
#             feats.append(x)
        
#         x = self.final_conv(x)
#         if is_feat and preact:
#             feats.append(x)
        
#         x = self.global_avg_pool(x)
#         x = x.view(x.size(0), -1)
        
#         if is_feat and not preact:
#             feats.append(x)
        
#         x = self.classifier(x)
        
#         if is_feat:
#             return feats, x
#         return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        ##pandora
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        out = x.mean(dim=(2, 3), keepdim=True)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return x * out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'mish':
            return x * torch.tanh(F.softplus(x))  # Mish activation function
        elif self.activation == 'hard_swish':
            return x * F.relu6(x + 3) / 6  # Hard Swish activation function
        else:
            return x

class HierarchicalSplitBlock(nn.Module):
    def __init__(self, channels):
        super(HierarchicalSplitBlock, self).__init__()
        self.split_channels = channels // 2
        self.path1 = ConvBlock(self.split_channels, self.split_channels, 3, padding=1, activation='mish')
        self.path2 = ConvBlock(self.split_channels, self.split_channels, 3, padding=1, activation='mish')
        ##pandora
        # self.path1 = ConvBlock(self.split_channels, self.split_channels * 2, 3, padding=1, activation='mish')
        # self.path2 = ConvBlock(self.split_channels, self.split_channels * 2, 3, padding=1, activation='mish')


    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.path1(x1)
        x2 = self.path2(x2)
        out = torch.cat([x1, x2], dim=1)
        return out

class BalancedLayer(nn.Module):
    def __init__(self, channels):
        super(BalancedLayer, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        weights = self.softmax(self.fc(x.mean(dim=(2, 3), keepdim=True))) # 计算通道权重
        return x * weights # 重新标定特征

class Hsnet(nn.Module):
    def __init__(self, num_classes):
        super(Hsnet, self).__init__()
        self.initial_conv = ConvBlock(3, 32, 3, padding=1)  # Initial channels初始通道
        self.hierarchical_block = HierarchicalSplitBlock(32)
        self.se_block = SEBlock(32)
        self.balanced_layer = BalancedLayer(32)
        self.final_conv = ConvBlock(32, 64, 1)  # Adjusted final channels
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, is_feat=False, preact=False,before=False):
        out = self.initial_conv(x)
        # print(f"After initial_conv: {out.shape}")  # 调试信息
        f0 = out

        out = self.hierarchical_block(out)
        # print(f"After hierarchical_block: {out.shape}")  # 调试信息
        f1 = out

        out = self.se_block(out)
        # print(f"After se_block: {out.shape}")  # 调试信息
        f2 = out

        out = self.balanced_layer(out)
        # print(f"After balanced_layer: {out.shape}")  # 调试信息
        f3 = out

        out = self.final_conv(out)
        # print(f"After final_conv: {out.shape}")  # 调试信息
        f4 = out

        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        # print(f"After global_avg_pool and view: {out.shape}")  # 调试信息
        f5 = out

        ##pandora
        # 确保输入特征形状匹配线性层的输入
        if f5.shape[1] != 64:
            # print(f"Adjusting input shape from {f5.shape} to (batch_size, 64)")  # 调试信息
            f5 = f5.view(f5.size(0), 64)  # 调整输入形状
            
        out = self.classifier(out)
        # print(f"After classifier: {out.shape}")  # 调试信息

        if is_feat:
            return [f0, f1, f2, f3, f4, f5], out
        else:
            return out


# # Example use of the network
# num_classes = 10  # Assuming 10 classes for this example
# model = Hsnet(num_classes)
# print(model)

# # Example input tensor of shape (1, 3, 32, 32)
# input_tensor = torch.rand(1, 3, 32, 32)

# # Forward pass through the network
# output = model(input_tensor)
# print("Output shape:", output.shape)  # Should be [1, num_classes] indicating the class scores
