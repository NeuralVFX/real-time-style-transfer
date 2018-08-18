import random
from torch.utils.data import *
from torchvision import transforms
import torch
import numpy as np
import cv2
import glob

cv2.setNumThreads(0)


############################################################################
#  Loader Utilities
############################################################################


class NormDenorm:
    # Store mean and std for transforms, apply normalization and de-normalization
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def norm(self, img, tensor=False):
        # normalize image to feed to network
        if tensor:
            return (img - float(self.mean[0])) / float(self.std[0])
        else:
            return (img - self.mean) / self.std

    def denorm(self, img, cpu=True, variable=True):
        # reverse normalization for viewing
        if cpu:
            img = img.cpu()
        if variable:
            img = img.data
        img = img.numpy().transpose(1, 2, 0)
        return img * self.std + self.mean


def cv2_open(fn):
    # Get image with cv2 and convert from bgr to rgb
    try:
        im = cv2.imread(str(fn), cv2.IMREAD_UNCHANGED + cv2.IMREAD_ANYDEPTH + cv2.IMREAD_ANYCOLOR).astype(
            np.float32) / 255
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f'Image Open Failure:{fn}  Error:{e}')


############################################################################
# Image Augmentation Stuff
############################################################################


def make_img_square(input_img):
    # Take rectangular image and crop to square
    height = input_img.shape[0]
    width = input_img.shape[1]
    if height > width:
        input_img = input_img[height // 2 - (width // 2):height // 2 + (width // 2), :, :]
    if width > height:
        input_img = input_img[:, width // 2 - (height // 2):width // 2 + (height // 2), :]
    return input_img


class FlipCV(object):
    # resize image and bbox
    def __init__(self, p_x=.5, p_y=.5):
        self.p_x = p_x
        self.p_y = p_y

    def __call__(self, sample):

        flip_x = self.p_x > random.random()
        flip_y = self.p_y > random.random()
        if not flip_x and not flip_y:
            return sample
        else:
            image = sample['image']
            if flip_x and not flip_y:
                image = cv2.flip(image, 1)
            if flip_y and not flip_x:
                image = cv2.flip(image, 0)
            if flip_x and flip_y:
                image = cv2.flip(image, -1)
            return {'image': image}


class ResizeCV(object):
    # resize image and bbox
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        image = make_img_square(image)
        image = cv2.resize(image, (self.output_size, self.output_size), interpolation=cv2.INTER_AREA)
        return {'image': image}


class TransformCV(object):
    # apply random transform

    def __init__(self, rot=.1, height=.1, width=.1, zoom=.2):
        # store range of possible transformations
        self.rot = rot
        self.height = height
        self.width = width
        self.zoom = zoom

    def get_random_transform(self, image):
        # create random transformation matrix
        rows, cols, ch = image.shape
        height = ((random.random() - .5) * 2) * self.width
        width = ((random.random() - .5) * 2) * self.height
        rot = ((random.random() - .5) * 2) * self.rot
        zoom = (random.random() * self.zoom) + 1

        rotation_matrix = cv2.getRotationMatrix2D((cols / 2,
                                                   rows / 2),
                                                  rot,
                                                  zoom)

        rotation_matrix = np.array([rotation_matrix[0],
                                    rotation_matrix[1],
                                    [0, 0, 1]])
        tx = width * cols
        ty = height * rows

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])

        transform_matrix = np.dot(translation_matrix, rotation_matrix)
        return transform_matrix

    def __call__(self, sample):
        # get transform and apply to both images
        image_a = sample['image']
        rows, cols, ch = image_a.shape
        transform_matrix = self.get_random_transform(image_a)

        image_a = cv2.warpAffine(image_a,
                                 transform_matrix[:2, :],
                                 (cols, rows),
                                 borderMode=cv2.BORDER_REFLECT,
                                 flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_AREA)

        return {'image': image_a}


############################################################################
#  Dataset and Loader
############################################################################


def open_style(style_path, transform, batch_size=8, size=256):
    # Load image and create stack based on batch size
    image = cv2_open(style_path)
    resize = ResizeCV(size)
    image = resize({'image': image})['image']
    image = torch.FloatTensor(np.rollaxis(transform.norm(image), 2))
    style_stack = []

    [style_stack.append(image) for a in range(batch_size)]
    style = torch.stack(style_stack)
    return style


class GenericDataset(Dataset):
    # Load Images from User Supplied Path and Apply Augmentation
    def get_train_and_test(self, data_perc, test_perc, seed=5):
        # use to only load a percentage of the training set
        total_count = len(self.path_list_a)
        ids = list(range(total_count))

        # seed this consistently even if model restarted
        np.random.seed(seed)

        ids = np.random.permutation(ids)
        split_index = int(total_count * data_perc)
        filtered_ids, _ = np.split(ids, [split_index])

        ids = np.random.permutation(filtered_ids)
        split_index = int(ids.size * test_perc)
        test_ids, train_ids = np.split(ids, [split_index])
        return train_ids, test_ids

    def __init__(self, path_a, transform, output_res=256, test_perc=.1, data_perc=1):
        self.train = True
        self.transform = transform
        self.path_list_a = glob.glob(f'{path_a}/*.*')
        self.output_res = output_res
        self.default_transform = NormDenorm([.5, .5, .5],
                                            [.5, .5, .5])
        self.data_transforms = transforms.Compose([ResizeCV(output_res * 2),
                                                   FlipCV(p_x=.5, p_y=0),
                                                   TransformCV(rot=15, height=.1, width=.1, zoom=.2),
                                                   ResizeCV(output_res)])

        self.data_transforms_test = transforms.Compose([ResizeCV(output_res)])
        self.train_ids, self.test_ids = self.get_train_and_test(data_perc, test_perc)

    def transform_set(self, image_a):
        # Apply augmentation
        trans_dict = {'image': image_a}
        if self.train:
            trans_dict = self.data_transforms(trans_dict)
        else:
            trans_dict = self.data_transforms_test(trans_dict)

        return np.rollaxis(self.transform.norm(trans_dict['image']), 2), np.rollaxis(
            self.default_transform.norm(trans_dict['image']), 2)

    def __getitem__(self, index):
        # lookup id from permuted list, apply transform, return tensor
        if self.train:
            lookup_id = self.train_ids[index]
        else:
            lookup_id = self.test_ids[index]

        image_path_a = self.path_list_a[lookup_id]
        image = cv2_open(image_path_a)

        image_vgg, image_default = self.transform_set(image)
        tensor_vgg = torch.FloatTensor(image_vgg)
        tensor_default = torch.FloatTensor(image_default)

        return tensor_vgg, tensor_default

    def __len__(self):
        if self.train:
            return self.train_ids.size
        else:
            return self.test_ids.size
