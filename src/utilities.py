import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from skimage.io import imsave
from skimage.exposure import adjust_gamma
from skimage.transform import resize, rotate


def create_validation_set(validation_file_paths, flip_original=False, folder_name=None):
    """
    This method randomly picks n_val_per_device train samples for each of the devices and creates validation samples for them.
    :param n_validation_files: The number of validation originals we want for each of the devices.
    :return: None. Creates validation samples and saves them to disk.
    """

    # get the test / validation split
    validation_files = validation_file_paths.copy()

    # create the output folders
    for directory in validation_files.folder_path.unique():
        directory = directory.replace("train", folder_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

    for file_name, img_path, folder_path in zip(validation_files.file_name, validation_files.file_path, validation_files.folder_path):
        # define the template for where we want to output the file
        folder_out_path = folder_path.replace("train", folder_name)
        file_name_prefix = file_name[:-4]

        # read and crop the image
        x = read_img(img_path)

        # if requested, flip the image
        if flip_original:
            x_flip = x[:, ::-1, :]

        # save the cropped original / transformed (with probability 0.5 in order to be as close to the test set as possible)
        if np.random.uniform() > 0.5:
            imsave(folder_out_path + file_name_prefix + "orig_unalt.jpg", crop_image(x, img_h=512, img_w=512), quality=100)
        else:
            pick_transformation(x, np.random.randint(0, 7+1), folder_out_path + file_name_prefix + "orig_manip.jpg", crop_random=False)

        # do the same thing for the flipped image, if requested
        if flip_original:
            if np.random.uniform() > 0.5:
                imsave(folder_out_path + file_name_prefix + "flip_unalt.jpg", crop_image(x_flip, img_h=512, img_w=512), quality=100)
            else:
                pick_transformation(x_flip, np.random.randint(0, 7 + 1), folder_out_path + file_name_prefix + "flip_manip.jpg", crop_random=False)


def create_train_set(train_file_paths, n_crops=3, flip_original=True, folder_name="train_augmented"):
    # get the training paths
    dt = train_file_paths.copy()

    for directory in dt.folder_path.unique():
        directory = directory.replace("train", folder_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

    for j, file_path in enumerate(dt.file_path):
        start_time = time.time()
        print("Transforming train image %d of %d." % (j, dt.shape[0]))
        out_path_base = file_path.replace("train", folder_name)[:-4]
        x = read_img(file_path)

        if flip_original:
            x_flip = x[:, ::-1, :]

        # create a list with integers from 0 to 7. Each integer corresponds to a transformation.
        # this list will be used to determine which transformation to perform on for each augmented sample we want to
        # create.
        choice_base = np.arange(0, 7 + 1)
        transform_numbers = np.random.choice(choice_base, size=2 * n_crops if flip_original else n_crops, replace=True)

        # create one central crop and (n_crops-1) random 512x512 crops of the original image
        # central crop
        imsave(out_path_base + "_orig_unalt_0.jpg", crop_image(x, img_h=512, img_w=512), quality=100)  # unaltered
        pick_transformation(x, transform_numbers[0], out_path_base + "_orig_manip_0.jpg", crop_random=False)  # manipulated

        # random crop
        for i in range(1, n_crops):
            imsave(out_path_base + "_orig_unalt_%d.jpg" % i, random_crop(x, img_h=512, img_w=512), quality=100)  # unaltered
            pick_transformation(x, transform_numbers[i], out_path_base + "_orig_manip_%d.jpg" % i, crop_random=True)  # manipulated

        # do the same for flipped images, if requested
        if flip_original:
            # central crop
            imsave(out_path_base + "_flip_unalt_%d.jpg" % n_crops, crop_image(x_flip, img_h=512, img_w=512), quality=100) # unaltered
            pick_transformation(x, transform_numbers[n_crops], out_path_base + "_flip_manip_%d.jpg" % n_crops, crop_random=False)  # manipulated

            # random crop
            for i in range(1, n_crops):
                t_i = n_crops + i
                imsave(out_path_base + "_flip_unalt_%d.jpg" % i, random_crop(x_flip, img_h=512, img_w=512), quality=100)  # unaltered
                pick_transformation(x, transform_numbers[t_i], out_path_base + "_flip_manip_%d.jpg" % i, crop_random=True)  # manipulated

        print("Completed train image %d of %d. Seconds elapsed for this sample: %.2f" % (j, dt.shape[0], time.time()-start_time))


def train_validation_split(n_validation_files, reuse_old_split=True, write_to_disk=False):
    # check if the files exist
    train_exists = os.path.exists(os.getcwd().replace("\\", "/") + "/data/train_files.csv")
    validation_exists = os.path.exists(os.getcwd().replace("\\", "/") + "/data/validation_files.csv")

    if reuse_old_split:
        if train_exists and validation_exists:
            return pd.read_csv(os.getcwd().replace("\\", "/") + "/data/train_files.csv"), \
                   pd.read_csv(os.getcwd().replace("\\", "/") + "/data/validation_files.csv")
        else:
            print("Old split not found / is incomplete. Creating a new split")
            write_to_disk = True

    # create a list that will be sampled from to decide which images will go into the validation set
    choice_base = np.arange(1, 275 + 1)

    # create lists to hold the train set data
    train_file_names = []
    train_file_paths = []
    train_file_folders = []
    train_file_classes = []

    # create lists to hold the validation set data
    validation_file_names = []
    validation_file_paths = []
    validation_file_folders = []
    validation_file_classes = []

    for k, folder in enumerate(os.listdir("./data/train/")):
        # file path base
        file_name_base = os.listdir("./data/train/" + folder)[0][:-5] + "%d.jpg"
        file_path_base = "./data/train/" + folder + "/" + file_name_base

        # randomly sample from the list of possible indices in this folder
        validation_indices = sorted(np.random.choice(choice_base, size=n_validation_files, replace=False))
        train_indices = set(choice_base).difference(validation_indices)

        # create the train file data
        train_file_names += [file_name_base % i for i in train_indices]
        train_file_paths += [file_path_base % i for i in train_indices]
        train_file_folders += ["./data/train/" + folder + "/"] * (275 - n_validation_files)
        train_file_classes += [k] * (275 - n_validation_files)

        # create the validation file data
        validation_file_names += [file_name_base % i for i in validation_indices]
        validation_file_paths += [file_path_base % i for i in validation_indices]
        validation_file_folders += ["./data/train/" + folder + "/"] * n_validation_files
        validation_file_classes += [k] * n_validation_files

    # create train DataFrame
    train = pd.DataFrame.from_dict(dict(
        file_name=train_file_names,
        file_path=train_file_paths,
        folder_path=train_file_folders,
        class_n=train_file_classes
    ))

    # create validation DataFrame
    validation = pd.DataFrame.from_dict(dict(
        file_name=validation_file_names,
        file_path=validation_file_paths,
        folder_path=validation_file_folders,
        class_n=validation_file_classes
    ))

    if write_to_disk:
        train.to_csv("./data/train_files.csv", index=False)
        validation.to_csv("./data/validation_files.csv", index=False)

    return train, validation


def pick_transformation(x, transform_num, out_path, crop_random):
    # crop the image
    x_cropped = random_crop(x) if crop_random else crop_image(x)

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
        imsave(out_path, resize_image(x, factor=0.5, crop_random=crop_random), quality=100)
    elif transform_num == 5:
        imsave(out_path, resize_image(x, factor=0.8, crop_random=crop_random), quality=100)
    elif transform_num == 6:
        imsave(out_path, resize_image(x, factor=1.5, crop_random=crop_random), quality=100)
    elif transform_num == 7:
        imsave(out_path, resize_image(x, factor=2, crop_random=crop_random), quality=100)


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
        out.append(read_img("temp_img.jpg"))

    os.remove("temp_img.jpg")

    # apply gamma correction
    for gamma in gamma_coefs:
        out.append(adjust_gamma(image=x_cropped, gamma=gamma))

    # resize the image using different factors
    for factor in resize_factors:
        out.append(resize_image(x=x, factor=factor, img_h=img_h, img_w=img_w))

    return out


def resize_image(x, factor, crop_random, img_h=512, img_w=512):
    resized_h = int(x.shape[0] * factor)
    resized_w = int(x.shape[1] * factor)

    if resized_h < img_h or resized_w < img_w:
        print("Can not resize image by given factor and still return an image of size %dx%d." % (img_h, img_w))
        # return randomly cropped image
        return random_crop(x, img_h=img_h, img_w=img_w)

    # if the factor is reasonable, resize the image
    x = resize(x, output_shape=(resized_h, resized_w), order=3, mode="constant")

    # return resized image
    if crop_random:
        return random_crop(x, img_h=img_h, img_w=img_w)
    else:
        return crop_image(x, img_h=img_h, img_w=img_w)


def crop_image(x, img_h=512, img_w=512, h_idx=None, w_idx=None):
    # get the indices at which the image should be cropped
    h = (x.shape[0] - img_h) // 2 if h_idx is None else h_idx
    w = (x.shape[1] - img_w) // 2 if w_idx is None else w_idx

    if h + img_h > x.shape[0]:
        raise ValueError("The starting index for height 'h_idx' is too large.")

    if w + img_w > x.shape[1]:
        raise ValueError("The starting index for width 'w_idx' is too large.")

    # return the cropped image
    return x[h: h + img_h, w:w + img_w, :]

def random_crop(x, img_h=512, img_w=512, buffer_size=10):
    h = np.random.randint(buffer_size, x.shape[0] - img_h - buffer_size)
    w = np.random.randint(buffer_size, x.shape[1] - img_w - buffer_size)

    return crop_image(x, img_h=img_h, img_w=img_w, h_idx=h, w_idx=w)

def read_img(img_path):
    """
    :param img_path: The path to the image that needs to be read.
    :return: The image in numpy.array format.

    The reason that a function is needed for this, is because the skimage.imread function returns a rotated image if
    the orientation of the image is not 1. This needs to be handled somehow.
    """
    img = Image.open(img_path)
    try:
        orientation = img._getexif()[274]
    except:
        return np.array(img)

    # create numpy array from img
    img = np.array(img)

    if orientation == 1:
        return img
    elif orientation == 3:
        return img[::-1, :, :]
    elif orientation == 6:
        return rotate(img, angle=-90, resize=True)
    elif orientation == 8:
        return rotate(img, angle=90, resize=True)
    else:
        raise ValueError("Unhandled orientation value: %d!" % orientation)
