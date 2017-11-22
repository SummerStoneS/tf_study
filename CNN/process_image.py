import glob
from collections import defaultdict
from itertools import groupby
import tensorflow as tf


image_filenames = glob.glob(".\\imagenet-dogs\\images\\n02*\\*.jpg")
training_dataset = defaultdict(list)
testing_dataset = defaultdict(list)

# [(狗名编号-狗名,狗图编号.jpg)]
image_filename_with_breed = map(lambda x: (x.split('\\')[3],x), image_filenames)

for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x: x[0]):
    # 根据list里每个tuple的第一个元素也就是狗品种做groupby,同属于一个品种的图片构成一个iterator
    for i, breed_image in enumerate(breed_images):

        # 遍历每个品种下的每一张图
        if i % 5 == 0:
            testing_dataset[dog_breed].append(breed_image[1])
        else:
            training_dataset[dog_breed].append(breed_image[1])

    # 检查每个品种的测试图像是否至少有全部图像的18%
    breed_training_count = len(training_dataset[dog_breed])
    breed_testing_count = len(testing_dataset[dog_breed])

    assert round(breed_testing_count/(breed_testing_count+breed_training_count), 2) > 0.18, 'not enough testing images'


"""
将图像转化为TFRecord文件
"""
def write_records_file()
