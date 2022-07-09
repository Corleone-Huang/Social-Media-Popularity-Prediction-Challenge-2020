# !/usr/bin/env python
# coding: utf-8
'''
@File    :   resnext101_finetune.py
@Time    :   2020/04/19 00:12:29
@Author  :   Wang Kai 
@Version :   1.0
@Contact :   wk15@mail.ustc.edu.cn
'''
# This is used to finetune ResNext101 model by Category or Subcategory

import argparse
import copy
import csv
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm, trange
from resnest.torch import resnest50,resnest101,resnest200,resnest269
# net = resnest50(pretrained=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,6,7"
random_seed=2020

train_image_root_path = "/home/wangkai/SMP/data/train/train_image/"
train_temporalspatial_filepath = "/home/wangkai/SMP/data/train/train_temporalspatial.json"
train_category_filepath= "/home/wangkai/SMP/data/train/train_category.json"
test_image_root_path = "/home/smp/SMP2020_test/SMP_test_images/"
test_temporalspatial_filepath = "/home/smp/SMP2020_test/test_temporalspatial.json"
test_category_filepath = "/home/smp/SMP2020_test/test_category.json"


def load_image(image_path):
    """ 
    load image data, convert image To Tenor
    """
    data_transforms = transforms.Compose([
        # transforms.Resize(224),
        transforms.Resize(416),
        # transforms.CenterCrop(224),
        transforms.CenterCrop(416),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    try:
        return data_transforms(Image.open(image_path).convert("RGB"))
    except:
        print("There is no file : {}".format(image_path))


class Image_Data(Dataset):
    def __init__(self, dataset):
        self.image_list = list(dataset['image_path'])
        self.uid = dataset["uid"]
        self.pid = dataset["pid"]
        self.category = dataset["category"]
        self.subcategory = dataset["subcategory"]
        print("uid:{} pid:{} image:{} ".format(len(self.uid), len(self.pid),
                                               len(self.image_list)))

    def __len__(self):
        return len(self.pid)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = load_image(image_path)
        return (self.pid[index], self.uid[index], image, self.category[index], self.subcategory[index])


def initialize_model(model_name="resnext101", finetune_by_category=True, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    num_classes = 11 if finetune_by_category else 77

    if model_name == "resnext101":
        """ ResNext101
        """
        model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "resnest269":
        """ ResNest269
        """
        model_ft = resnest269(pretrained=False)
        model_ft.load_state_dict(torch.load("/home/wangkai/pretrain_model/resnest269-0cc87c48.pth"))
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 416

    elif model_name == "resnext50":
        """ ResNext50
        """
        model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "resnet152":
        """ ResNet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def crete_dataset():
    """ 
    create image dataset
    """
    df_train_temporalspatial = pd.read_json(train_temporalspatial_filepath)
    df_test_temporalspatial = pd.read_json(test_temporalspatial_filepath)
    df_train_category = pd.read_json(train_category_filepath)
    df_test_category = pd.read_json(test_category_filepath)

    df_train=pd.merge(df_train_category,df_train_temporalspatial)
    df_train["Imagepath"]=train_image_root_path + df_train["Uid"] + "/" + df_train["Pid"].apply(str) + ".jpg"
    df_test=pd.merge(df_test_category,df_test_temporalspatial)
    df_test["Imagepath"]=test_image_root_path + df_test["Uid"] + "/" + df_test["Pid"].apply(str) + ".jpg"
    df=pd.concat([df_train,df_test],axis=0)

    # encode the category and subcategory to number
    # 11 Category number
    df["Category_code"] = df["Category"].astype("category").cat.codes
    # 77 Subcategory number
    df["Subcategory_code"] = df["Subcategory"].astype("category").cat.codes

    data = {
        "uid": np.array(df["Uid"]),
        "pid": np.array(df["Pid"]),
        "image_path": np.array(df["Imagepath"]),
        "category": np.array(df["Category_code"]),
        "subcategory": np.array(df["Subcategory_code"])
    }
    return data


def train_model(model, args, data_loader, criterion, optimizer, scheduler):
    """
    Support function for model training.
    Args:
      model: Model to be trained
      criterion: Optimization criterion (loss)
      optimizer: Optimizer to use for training
      num_epochs: Number of epochs
      use_gpu: default is True (Bool)
     """
    num_epochs = args.num_epochs
    use_gpu = args.use_gpu
    finetune_by_category = args.finetune_by_category

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if use_gpu == True:
        model.to("cuda")
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    k = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        train_loss = 0.0
        model.train()
        train_corrects = 0.0
        train_total = 0
        # Iterate over data.
        for _, (pids, uids, inputs, categorys, subcategorys) in tqdm(enumerate(data_loader)):
            inputs = inputs.to("cuda") if use_gpu else inputs
            if finetune_by_category:
                labels = categorys.to("cuda",dtype=torch.int64) if use_gpu else categorys
            else:
                labels = subcategorys.to("cuda",dtype=torch.int64) if use_gpu else subcategorys
            # labels = labels.view(-1,1).to("cuda") if use_gpu else labels.view(-1, 1)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()
            scheduler.step()
            # statistics
            train_total += inputs.size(0)
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
            print("---epoch:{} train Loss: {:.4f} acc: {:.4f} top5:".format(epoch,loss.item(), torch.sum(preds == labels.data).item()/inputs.size(0)),preds[:5])

        epoch_train_loss = train_loss / train_total
        epoch_train_corrects = train_corrects/train_total
        print("train Loss: {:.4f} Acc: {:.4f}".format(
            epoch_train_loss, epoch_train_corrects))


    time_elapsed = time.time() - since
    print('Finetune complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
   
    # load best model weights
   
    return model


def extract_feature(model, args, data_loader, use_gpu=True, finetune_by_category=False):

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_ftrs)
    torch.nn.init.eye_(model.fc.weight)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if args.use_gpu == True:
        model.to("cuda")

    model.eval()
    if finetune_by_category:
        image_feature_filepath = args.finetune_category_feature_filepath
    else:
        image_feature_filepath = args.finetune_subcategory_feature_filepath
    # image_feature_filepath= "/home/smp/SMP_test_feature/Pretrained_224ResNext101_image.csv"
    with open(image_feature_filepath, mode="w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        columns = ["pid", "uid"] + \
            ["image_feature"+str(i+1) for i in range(2048)]
        writer.writerow(columns)
        for _, (pids, uids, images, categorys, subcategorys) in tqdm(enumerate(data_loader)):
            images = images.to("cuda") if use_gpu else images
            features = model(images)
            features = features.to("cpu").data.numpy()
            pids = pids.numpy().reshape(-1, 1)
            uids = np.array(uids).reshape(-1, 1)
            writer.writerows(np.concatenate(
                (pids, uids, features), axis=1).tolist())


def main(args):

    data = crete_dataset()
    print("Initializing Datasets and Dataloaders...")
    dataset = Image_Data(data)
    data_loader = DataLoader(dataset=dataset,batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    print('Start building finetune model')
    model, input_size = initialize_model(
        model_name=args.model, finetune_by_category=args.finetune_by_category, use_pretrained=args.use_pretrained)

    for item in list(model.children())[:-3]:
        for param in item.parameters():
            param.requires_grad=False
            
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        list(model.parameters())), lr=args.learning_rate)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, list(model.parameters())),
    #                             lr=args.learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=6, gamma=0.3)

    print("Start training model")
    model = train_model(model, args, data_loader,criterion=criterion, optimizer=optimizer, scheduler=scheduler)
    torch.save(model, "/home/wangkai/SMP/checkpoint/ResNext101_best_subcategory.pth")

    # print("extract finetuned feature...")
    # if isinstance(model, torch.nn.DataParallel):
    #     model = model.module
    # model.to("cpu")

    # extract_feature(model, args, data_loader=data_loader, use_gpu=args.use_gpu, finetune_by_category=args.finetune_by_category)
    # print("Over! extract finetuned feature.\n")
    print("The process is over")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", type=bool, default=True,
                        help="will use gpu(default: True)")
    parser.add_argument("--model", type=str,
                        default="resnext101",
                        choices=["resnext101", "resnext50", "resnet152", "resnest269"],
                        help="model name")
    parser.add_argument("--finetune_category_feature_filepath", type=str,
                        default="/home/wangkai/SMP/feature/all_feature/Category_ResNext101_image_486194.csv",
                        help="image feature filepath extract by finetune category model")
    parser.add_argument("--finetune_subcategory_feature_filepath", type=str,
                        default="/home/wangkai/SMP/feature/all_feature/Subcategory_ResNext101_image_486194.csv",
                        help="image feature filepath extract by finetune subcategory model")
    parser.add_argument("--use_pretrained", type=bool, default=True,
                        help="use pretrain weight in ImageNet(default: True)")
    parser.add_argument("--finetune_by_category",
                        type=bool, default=False,
                        help="is finetuned by Category(default: False)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="batch size(default: 256)")
    parser.add_argument("--learning_rate", type=float,
                        default=0.0001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="number epochs(default: 1)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="number workers(default: 8)")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_arguments()
    main(args)
    pass
