import os
import math
from datetime import datetime
import sys
from collections import defaultdict

import torch
import torch.optim as optim
import numpy as np
import cv2

from model import UNet, train, test, two_class_calculate_bayes_errors


def write_results_to_file(results_file, results_dir, results_filename):
    with open(os.path.abspath(os.path.join(results_dir,results_filename+".tsv")),'w') as f:
        for result in results_file:
            for key,value in result.items():
                if key == "epoch_losses":
                    f.write(f"{key}\t"+"\t".join(str(a) for a in value)+"\n")
                elif key == "model_full":
                    torch.save(value,os.path.abspath(os.path.join(results_dir,results_filename+"_full_model_pt.pt")))
                elif key == "error_thresholds":
                    np.save(os.path.abspath(os.path.join(results_dir,results_filename+"_error_thresholds_npy.npy")),value)
                elif key == "test_metrics":
                    merged_items = defaultdict(list)
                    for val in value:
                        for k,v in val.items():
                           merged_items[k].append(v)
                    for k,v in merged_items.items():
                        f.write(f"{key}\t{k}\t"+"\t".join(str(a) for a in v)+"\n")
                else:
                    f.write(f"{key}\t{str(value)}\n")

if __name__ == "__main__":
    print("Begin program...")
    start = datetime.now()
    print(start)

    RESULTS_DIR = "results/"
    RESULTS_FILENAME = datetime.now().strftime("%Y_%m_%d__%H_%M")
    results_file = []

    DATA_DIR = "data/fluocells-resized1/"
    results_file.append({"DATA_DIR": DATA_DIR})
    IMAGES = "all_images/images/"
    MASKS = "all_masks/masks/"

    image_names = []  # image names are also mask names in another directory
    images_directory = os.fsencode(os.path.join(DATA_DIR,IMAGES))
    for file in os.listdir(images_directory):
        filename = os.fsdecode(file)
        if filename != ".gitkeep":
            image_names.append(filename)

    print("Loading data into memory begin...")
    # load data into memory
    #   0 is white 255 is black
    #   for this fluocells data, the images are full color.
    #   for this fluocells data, the masks are black and white where white is the target class.
    image_data = []
    for name in image_names:
        img = cv2.imread(os.path.join(DATA_DIR,os.path.join(IMAGES,name)))
        img_b, img_g, img_r = cv2.split(img)
        img = np.array([np.array(img_b).astype(np.float32),np.array(img_g).astype(np.float32),np.array(img_r).astype(np.float32)])
        img = img / 255.  # scale between 0 and 1

        msk = cv2.imread(os.path.join(DATA_DIR,os.path.join(MASKS,name)))
        msk_b, msk_g, msk_r = cv2.split(msk)
        msk = np.array([np.sum(np.array([np.array(msk_b).astype(np.float32),np.array(msk_g).astype(np.float32),np.array(msk_r).astype(np.float32)]), axis=0)])
        msk[msk > 0.] = 1. # scale as 0 or 1, assuming 1 if not all channels are 0
        tmp = np.copy(msk).astype(np.int32)
        tmp = np.bitwise_xor(tmp,np.ones(tmp.shape).astype(np.int32))
        # a mask has two channels. the 0th indexed channel has value 1 if white(target class) and 0 if black(negative class).
        #   the 1st indexed channel has value 1 if black(negative class) and 0 if white(target class).
        #   this effectively results in a matrix the size of the input images of 2D tensors such that [x,y] where x!=y and x,y are either 0 or 1.
        #     this is effectively representing a positive and negative class (two classes) such that a positive class's pixel is of value [1,0] and a negative class's pixel is of value [0,1].
        #     i am using two classes for this binary classification because of how Bayes error (maximum penultimate posterior amongst all classes) needs to be calculated.
        msk = np.concatenate([tmp,msk]).astype(np.float32)  # positive class value is 1 even though its value is 0

        image_data.append((img,msk))
    print("Loading data into memory end.")

    print("Test train split begin...")
    train_percentage = 0.8
    results_file.append({"train_percentage": train_percentage})
    train_data = image_data[:math.floor(train_percentage*len(image_data))]
    test_data = image_data[math.floor(train_percentage*len(image_data)):]
    print(f"Train data len is {len(train_data)}, test data len is {len(test_data)} at a train percentage of {train_percentage}.")
    print("Test train split end.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute device is: {'cuda' if torch.cuda.is_available() else 'cpu'}.")

    # constants
    input_dim = 3  # number of channels/layers to the image is the input dimension
    results_file.append({"input_dim": input_dim})
    output_dim = 2  # number of channels/layers for the output dimension, two for this binary classification because of the need for Bayes error calculation.
    results_file.append({"output_dim": output_dim})
    batch_size = 2  # this is the largest I could get with a g4ad.xlarge AWS EC2
    results_file.append({"batch_size": batch_size})
    learning_rate = 0.001
    results_file.append({"learning_rate": learning_rate})
    num_epochs = 10
    results_file.append({"num_epochs": num_epochs})
    dropout_rate = 0.7
    results_file.append({"dropout_rate": dropout_rate})
    total_pixels = image_data[0][0].shape[1]*image_data[0][0].shape[2]  # uniformly sized images
    num_samples = math.floor(total_pixels*(1-dropout_rate))  # dropout rate of X, so keep (1-X)
    print(f"Number of samples to keep is {num_samples} out of {total_pixels} at a dropout rate of {dropout_rate}.")

    # model
    model = UNet(in_channels=input_dim, out_channels=output_dim).to(device)
    print(str(model))
    results_file.append({"model_description": str(model)})
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    results_file.append({"optimizer": "Adam"})
    
    # train model
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    print("Beginning training...")
    epoch_losses = []
    for epoch in range(num_epochs):
        epoch_losses.append(train(model, train_loader, optimizer, num_samples, epoch))
        print(f"Epoch {epoch+1}/{num_epochs} complete.")
    results_file.append({"epoch_losses": epoch_losses})
    print("Training end.")
    results_file.append({"model_full": model})

    # calculate Bayes errors
    print("Calculating Bayes errors...")
    error_thresholds = np.zeros([1,image_data[0][0].shape[1], image_data[0][0].shape[2]])
    error_thresholds = two_class_calculate_bayes_errors(model, train_loader, num_samples, error_thresholds)
    results_file.append({"error_thresholds": error_thresholds})
    print("End calculating Bayes errors.")

    # test model
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)  # I want to do this one test image at a time
    print("Beginning testing...")
    test_metrics = test(model, test_loader, num_samples, error_thresholds)
    results_file.append({"test_metrics": test_metrics})
    print("Testing end.")

    results_file.append({"elapsed_time": str(datetime.now()-start)})
    results_file.append({"end_time": str(datetime.now())})

    write_results_to_file(results_file, RESULTS_DIR, RESULTS_FILENAME)

    print(f"Elapside runtime: {datetime.now() - start}.")
    print(datetime.now())
    print("End program.")

