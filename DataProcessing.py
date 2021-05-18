import glob
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import os

Anim_Image_Paths = glob.glob("D:\\files\\course\\DL\\Project\\data\\*.jpg")
Image_Detect_Path = "./ImageProcessing/lbpcascade_animeface/lbpcascade_animeface.xml"
Image_Paths = glob.glob("D:\\files\\course\\DL\\Project\\Anime_Face\\data\\anim_face\\anim_face\\*.jpg")

def MergeImages(images, row_size, col_size, savepath):
    width_i = 96
    height_i = 96

    line_max = row_size
    row_max = col_size

    num = 0
    pic_max = line_max * row_max

    toImage = Image.new('RGBA', (width_i * line_max, height_i * row_max))

    for i in range(0, row_max):
        for j in range(0, line_max):
            pic_fole_head = images[num]
            width, height = pic_fole_head.size

            tmppic = pic_fole_head.resize((width_i, height_i))

            loc = (int(i % line_max * width_i), int(j % line_max * height_i))

            toImage.paste(tmppic, loc)
            num = num + 1

        if num >= pic_max:
            break

    print(toImage.size)
    toImage.save(savepath)


def detect(image, cascade_file):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return image


if __name__ == "__main__":
    images = []
    merge_save_path = './data/dataprocessing/merged.png'
    for i in range(25):
        images.append(Image.open(Anim_Image_Paths[i]))
    MergeImages(images, 5, 5, merge_save_path)

    detect_save_path = './data/dataprocessing/detect.png'
    detect_images = []
    for i in range(25):
        img = cv2.imread(Anim_Image_Paths[i+25])
        detect_image = detect(img, Image_Detect_Path)
        detect_image = Image.fromarray(detect_image)
        detect_images.append(detect_image)
    MergeImages(detect_images, 5, 5, detect_save_path)

    resize_save_path = './data/dataprocessing/resize.png'
    resize_images = []
    for i in range(25):
        resize_images.append(Image.open(Image_Paths[i]))
    MergeImages(resize_images, 5, 5, resize_save_path)