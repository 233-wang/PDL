import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from par_DW_LSTM4 import ParDWLSTM


def main():
    batch_size = 32
    num_classes = 5
    epochs = 30
    input_size = 9
    hidden_size = 32
    num_layers = 2

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # transform = transforms.Compose([transforms.RandomResizedCrop(16),
    #                                 transforms.RandomHorizontalFlip(),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path
    image_path = os.path.join(data_root, "LPN", "data_sequential_img")  # flower data set path
    print(image_path)
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=transform)
    train_num = len(train_dataset)
    print(train_num)
    
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=transform)
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # create model
    net = ParDWLSTM(input_size=input_size, hidden_size=hidden_size,
                  num_layers=num_layers, num_classes=num_classes, init_weights=True)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    best_acc = 0.0

    save_path = 'weight_res_seq/ParDWLSTM7_20_.pth'
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        valing_loss = 0.0
        train_acc = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images)
            predict_y = torch.max(logits, dim=1)[1]
            train_acc += torch.eq(predict_y, labels).sum().item()
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.4f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)


        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images)
                val_loss = loss_function(outputs, val_labels)
                # loss = loss_function(logits, labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels).sum().item()
                valing_loss += val_loss.item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)


        train_accurate = train_acc / train_num
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.4f  train_acc: %.4f  val_loss: %.4f  val_accuracy: %.4f' %
              (epoch + 1, running_loss / train_steps, train_accurate, valing_loss / val_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')



if __name__ == '__main__':
    main()