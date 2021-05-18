import torch
from torch.utils.data import Dataset
import h5py
from PIL import Image
import cv2
import numpy as np
import glob

class CACDDataset(Dataset):
    def __init__(self, dataset_path, transform, residual_path=None):
        super(CACDDataset, self).__init__()
        self.dataset_path = dataset_path
        with h5py.File(dataset_path, 'r') as file:
            self.length = len(file["origin_images"])
        self.transform = transform
        self.residual_path = residual_path

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        with h5py.File(self.dataset_path, 'r') as file1:
            image = file1["origin_images"][item]
        #image = np.resize(image, (96, 96, 3))
        #image = Image.fromarray(image)
        input_img = self.transform(image)
        return input_img


def Image_To_H5py(image_paths):
    file_name = "./data/Image_dataset"
    f = h5py.File(file_name, "w")
    print("image nums:".format(len(image_paths)))
    dataset = f.create_dataset("origin_images", (len(image_paths), 96, 96, 3))
    for i in range(len(image_paths)):
        img = Image.open(image_paths[i])
        img = np.array(img)
        norm_img = np.zeros(img.shape, dtype=np.float32)
        cv2.normalize(img, norm_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        if i % 1000 == 0:
            print("image nums:" + str(i))
        dataset[i] = norm_img

def Get_Data(path):
    f = h5py.File(path + "Image_dataset", "r")
    origin_images = f["origin_images"][:]
    images = torch.tensor(origin_images)
    print(images.size())
    return [images]


Anim_Image_Paths = glob.glob("D:\\files\\course\\DL\\Project\\Anime_Face\\data\\anim_face\\anim_face\\*.jpg")

if __name__ == "__main__":
    Image_To_H5py(Anim_Image_Paths)
    file_name = "./data/Image_dataset"
    f = h5py.File(file_name, "r")
    images = f["origin_images"]
    print(len(images))
