import random as rand
import sys
import math
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.down1 = DoubleConv(in_channels, 4)
        self.down2 = DoubleConv(4, 2)
        self.up2 = nn.ConvTranspose2d(2, 4, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(4*2, out_channels, 2, stride=2)  # '*2' because of the crossover
        
        self.out_channels = out_channels

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x = self.up2(x2)
        x = F.max_pool2d(x,2)
        x = self.up1(torch.cat((x, x1), 1))
        x = F.max_pool2d(x,2)
        return torch.sigmoid(x)


def dropout(source, destination, num_samples):
    indices = np.random.choice(np.arange(source.shape[2]*source.shape[3]), num_samples, replace=False)
    indices = [(math.floor(y/source.shape[3]),y%source.shape[3]) for y in indices]

    for i in range(destination.shape[0]):
        for j in range(destination.shape[1]):
            for index in indices:
                destination[i,j,index[0],index[1]] = source[i,j,index[0],index[1]]  # different channels may have different values
    
    return destination

def mc_dropout_stage(model, input, num_samples):
    raw = model(input)
    output = torch.zeros(raw.shape[0], raw.shape[1], raw.shape[2], raw.shape[3])
    output = output.cuda()

    output = dropout(raw, output, num_samples)

    return output

def train(model, train_loader, optimizer, num_samples, epoch):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    running_loss = 0.0
    epoch_start = datetime.now()
    for i, data in enumerate(train_loader):
        start = datetime.now()
        input, label = data
        input, label = input.cuda(), label.cuda()
        optimizer.zero_grad()
        outputs = mc_dropout_stage(model, input, num_samples)
        loss = loss_fn(outputs[:], label[:])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(f"Epoch: {epoch+1}, datum: {i+1}/{len(train_loader)}, elapsed time: {datetime.now()-start}")
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Loss: {epoch_loss} Elapsed time: {datetime.now()-epoch_start}")
    return epoch_loss

def two_class_calculate_bayes_errors(model, train_loader, num_samples, error_thresholds):
    for i, data in enumerate(train_loader):
        input, label = data
        input, label = input.cuda(), label.cuda()
        input_mcdropout = torch.zeros(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        input_mcdropout = input_mcdropout.cuda()
        input_mcdropout = dropout(input, input_mcdropout, num_samples)
        predictions = model(input_mcdropout)
        predictions = predictions.cpu()
        predictions = predictions.detach().numpy()
        # maximum penultimate posterior, per pixel, between each class
        minimums = np.fmin.reduce(predictions, axis=1)
        minimums = np.fmin.reduce(minimums)
        error_thresholds = np.fmax(minimums,error_thresholds)
    return error_thresholds

def two_class_decision_rules(output, error_thresholds):
    output = output.cpu()
    output = output.detach().numpy()
    classification = np.argmax(output, axis=1)
    classification[classification < error_thresholds] = -1  # -1 means reject action
    return classification 

def two_class_dice_similarity_score(output, label):
    label = label.cpu()
    label = label.detach().numpy()
    label = np.argmax(label, axis=1) # keep the index value which is greatest [1,0] means positive class and argmax's to 0(positive class value), [0,1] means negative class and argmax's to 1(negative class value)
    # 0 is positive 1 is negative, -1 is reject
    tp = np.sum([True for x,y in zip(output[0].flatten(),label[0].flatten()) if x == y and x == 0])
    tn = np.sum([True for x,y in zip(output[0].flatten(),label[0].flatten()) if x == y and x == 1])
    fp = np.sum([True for x,y in zip(output[0].flatten(),label[0].flatten()) if x != y and x == 0])
    fn = np.sum([True for x,y in zip(output[0].flatten(),label[0].flatten()) if x != y and x == 1])
    total_reject = np.sum([True for x in output[0].flatten() if x == -1])
    dice = np.sum([True for x,y in zip(output[0].flatten(),label[0].flatten()) if x == y])  # count how many are similar
    return dice, tp, tn, fp, fn, total_reject

def test(model, test_loader, num_samples, error_thresholds):
    test_metrics = []
    for i, data in enumerate(test_loader):
        input, label = data
        input, label = input.cuda(), label.cuda()
        input_mcdropout = torch.zeros(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        input_mcdropout = input_mcdropout.cuda()
        input_mcdropout = dropout(input, input_mcdropout, num_samples)
        predictions = model(input_mcdropout)
        output = two_class_decision_rules(predictions, error_thresholds)
        dice, tp, tn, fp, fn, total_reject = two_class_dice_similarity_score(output,label)
        print(f"Datum {i+1}/{len(test_loader)} with dice similarity scores\n(percentage of total pixels): {dice/(input.shape[2]*input.shape[3])},\n(percentage of non-reject pixels): {dice/((input.shape[2]*input.shape[3])-total_reject)}.")
        metric = {"dice": dice, "image_total_pixels": (input.shape[2]*input.shape[3]),
                  "tp": tp, "tn": tn, "fp": fp, "fn": fn, "total_reject": total_reject}
        test_metrics.append(metric)
    return test_metrics
