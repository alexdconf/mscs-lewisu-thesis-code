import os

import cv2

if __name__ == "__main__":
    print("Begin")
    # get the data
    DATA_DIR = "fluocells/"
    IMAGES = "all_images/images/"
    MASKS = "all_masks/masks/"

    DEST_DIR = "fluocells-resized1/"

    image_names = []  # image names are also mask names in another directory
    images_directory = os.fsencode(os.path.join(DATA_DIR,IMAGES))
    for file in os.listdir(images_directory):
        filename = os.fsdecode(file)
        image_names.append(filename)

    # load data into memory
    image_data = []
    for name in image_names:
        img = cv2.imread(os.path.join(DATA_DIR,os.path.join(IMAGES,name)))
        resized_img = cv2.resize(img, None, fx=0.25, fy=0.25)
        cv2.imwrite(os.path.join(DEST_DIR,os.path.join(IMAGES,"resized1_"+name)),resized_img)

        msk = cv2.imread(os.path.join(DATA_DIR,os.path.join(MASKS,name)))
        resized_msk = cv2.resize(msk, None, fx=0.25, fy=0.25)
        cv2.imwrite(os.path.join(DEST_DIR,os.path.join(MASKS,"resized1_"+name)),resized_msk)
    print("Done.")
