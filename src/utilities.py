import os
import time
import numpy as np
import pandas as pd
from skimage.io import imread, imsave
from skimage.exposure import adjust_gamma
from skimage.transform import resize


def create_validation_set(n_val_per_device=75):
    """
    This method randomly picks n_val_per_device train samples for each of the devices and creates validation samples for them.
    :param n_val_per_device: The number of validation samples we want for each of the devices.
    :return: None. Creates validation samples and saves them to disk.
    """

    # create a list that will be sampled from to decide which images will go into the validation set
    base = np.arange(1, 275 + 1)

    # create lists to hold the indices of the files that won't be in the validation data
    train_file_paths = []
    classes = []

    for k, folder in enumerate(os.listdir("./data/train/")):
        ind = np.random.choice(base, size=n_val_per_device, replace=False)
        prefix = os.listdir("./data/train/%s/" % folder)[0][:-5]
        j = 1

        # create the placeholder for where the generated samples should be saved
        in_placeholder = "./data/train/%s/%s%%d.jpg" % (folder, prefix)
        out_unaltered = "./data/validation/%s/%s%%d_unalt.jpg" % (folder, prefix)
        out_manipulated = "./data/validation/%s/%s%%d_manip.jpg" % (folder, prefix)

        # append the file and folder names for the training data that wasn't chosen to be in the validation set
        train_file_paths += [in_placeholder % ii for ii in list(set(base).difference(ind))]
        classes += ([k] * (275 - n_val_per_device))

        # check if the folder exists
        if not os.path.exists("./data/validation/%s/" % folder):
            os.makedirs("./data/validation/%s/" % folder)

        for i in ind:
            # read and crop the image
            img_path = in_placeholder % i
            x = imread(img_path)
            # the images from the folder Sony-NEX-7 return the image and info. We only want the image
            if folder == "Sony-NEX-7":
                x = x[0]
            x_cropped = crop_image(x)

            # save the cropped original
            imsave(out_unaltered % j, x_cropped, quality=100)
            imsave(out_unaltered % j, x_cropped, quality=100)
            j += 1

            #
            # compression
            # quality = 70
            imsave(out_manipulated % j, x_cropped, quality=70)
            j += 1
            # quality = 90
            imsave(out_manipulated % j, x_cropped, quality=90)
            j += 1

            #
            # gamma adjustment
            # gamma = 0.8
            imsave(out_manipulated % j, adjust_gamma(x_cropped, gamma=0.8))
            j += 1
            # gamma = 1.2
            imsave(out_manipulated % j, adjust_gamma(x_cropped, gamma=1.2))
            j += 1

            #
            # resize
            # factor = 0.5
            imsave(out_manipulated % j, resize_image(x, factor=0.5))
            j += 1
            # factor = 0.8
            imsave(out_manipulated % j, resize_image(x, factor=0.8))
            j += 1
            # factor = 1.5
            imsave(out_manipulated % j, resize_image(x, factor=1.5))
            j += 1
            # factor = 2
            imsave(out_manipulated % j, resize_image(x, factor=2))
            j += 1

    pd.DataFrame.from_dict({
        "file_path": train_file_paths,
        "class": classes
    }).to_csv("./data/train_data_classes.csv", index=False)

def create_train_set(train_data_classes):
    dt = train_data_classes.copy()
    dt["directory"] = dt.file_path.apply(lambda x: "/".join((x.replace("train", "train_aug")[:-4]).split('/')[:-1]) + "/")
    choice_base = np.arange(0, 7 + 1)

    for directory in dt.directory.unique():
        if not os.path.exists(directory):
            os.makedirs(directory)

    for j, file_path in enumerate(dt.file_path):
        start_time = time.time()
        print("Transforming train image %d of %d." % (j, dt.shape[0]))
        out_path_base = file_path.replace("train", "train_aug")[:-4]
        x = imread(file_path)

        if "Sony-NEX-7" in file_path:
            x = x[0]
        x_flip = x[:, ::-1, :]

        # save the original and its flipped as 'unaltered' after cropping
        x_cropped = crop_image(x, img_w=512, img_h=512)
        x_flip_cropped = crop_image(x_flip, img_w=512, img_h=512)
        imsave(out_path_base + "_unalt_0.jpg", x_cropped, quality=100)
        imsave(out_path_base + "_unalt_1.jpg", x_flip_cropped, quality=100)

        print("Transforming the images.")
        # get the transformations of the two images
        transform_numbers = np.random.choice(choice_base, 4, replace=False)
        pick_transformation(x, transform_numbers[0], out_path_base + "_manip_0.jpg")
        pick_transformation(x, transform_numbers[1], out_path_base + "_manip_1.jpg")
        pick_transformation(x_flip, transform_numbers[2], out_path_base + "_manip_2.jpg")
        pick_transformation(x_flip, transform_numbers[3], out_path_base + "_manip_3.jpg")
        print("Completed train image %d of %d. Seconds elapsed for this sample: %.2f" % (j, dt.shape[0], time.time()-start_time))

def pick_transformation(x, transform_num, out_path):
    # crop the image
    x_cropped = crop_image(x)

    # choose one of the 7 transformations to apply to the image
    if transform_num == 0:
        imsave(out_path, x_cropped, quality=70)
    elif transform_num == 1:
        imsave(out_path, x_cropped, quality=90)
    elif transform_num == 2:
        imsave(out_path, adjust_gamma(x_cropped, gamma=0.8), quality=100)
    elif transform_num == 3:
        imsave(out_path, adjust_gamma(x_cropped, gamma=1.2), quality=100)
    elif transform_num == 4:
        imsave(out_path, resize_image(x, factor=0.5), quality=100)
    elif transform_num == 5:
        imsave(out_path, resize_image(x, factor=0.8), quality=100)
    elif transform_num == 6:
        imsave(out_path, resize_image(x, factor=1.5), quality=100)
    elif transform_num == 7:
        imsave(out_path, resize_image(x, factor=2), quality=100)


def transform_images(v, img_h=512, img_w=512, compression_coefs=[100, 90, 70], gamma_coefs=[0.8, 1.2], resize_factors=[0.5, 0.8, 1.5, 2]):
    out = []
    for transformations in [transform_image(x, img_h, img_w, compression_coefs, gamma_coefs, resize_factors) for x in v]:
        for img in transformations:
            out.append(img)

    return out


def transform_image(x, img_h, img_w, compression_coefs, gamma_coefs, resize_factors):
    out = []

    # crop the image
    x_cropped = crop_image(x, img_h, img_w)

    # compress the image (quality = 100 is the original)
    # DEVNOTE: saving the file to disk is a hack. I'll see if there's a way around this later.
    for quality in compression_coefs:
        imsave(fname="temp_img.jpg", arr=x_cropped, quality=quality)
        out.append(imread("temp_img.jpg"))

    os.remove("temp_img.jpg")

    # apply gamma correction
    for gamma in gamma_coefs:
        out.append(adjust_gamma(image=x_cropped, gamma=gamma))

    # resize the image using different factors
    for factor in resize_factors:
        out.append(resize_image(x=x, factor=factor, img_h=img_h, img_w=img_w))

    return out


def resize_image(x, factor, img_h=512, img_w=512):
    resized_h = int(x.shape[0] * factor)
    resized_w = int(x.shape[1] * factor)

    if resized_h < img_h or resized_w < img_w:
        raise ValueError("Can not resize image by given factor and still return an image of size %dx%d." % (img_h, img_w))

    # if the factor is reasonable, resize the image
    x = resize(x, output_shape=(resized_h, resized_w), order=3, mode="constant")

    # return resized image
    return crop_image(x, img_h=img_h, img_w=img_w)


def crop_image(x, img_h=512, img_w=512):
    # get the indices at which the image should be cropped
    h = (x.shape[0] - img_h) // 2
    v = (x.shape[1] - img_w) // 2

    # return the cropped image
    return x[h: h + img_h, v:v + img_w, :]
