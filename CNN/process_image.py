import glob
from collections import defaultdict
from itertools import groupby
import tensorflow as tf

sess = tf.Session()

image_filenames = glob.glob(".\\imagenet-dogs\\images\\n02*\\*.jpg")
training_dataset = defaultdict(list)
testing_dataset = defaultdict(list)

# [(狗名编号-狗名,狗图编号.jpg)]
image_filename_with_breed = map(lambda x: (x.split('\\')[3], x), image_filenames)

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
将图像转化为TFRecord文件,y和train_X一一对应
"""


def write_records_file(dataset, record_location):
    """
    :param dataset: 字典类型，key=狗品种，value：该品种狗的jpg路径 list
    :param record_location: str，存储TFRecord输出的路径
    :return:
    """
    writer = None
    current_index = 0
    for breed, images_filenames in dataset.items():
        for image_filename in image_filenames:
            if current_index % 100 == 0:
                # 每100张图才放到TFRecord文件里，加速
                if writer:
                    writer.close()

                record_filename = "{record_location}-{current_index}.tfrecords".format(record_location=record_location,
                                                                                       current_index=current_index)
                writer = tf.python_io.TFRecordWriter(record_filename)           # 初始化writer
                current_index += 1
                image_file = tf.read_file(image_filename)
                try:
                    image = tf.image.decode_jpeg(image_file)                    # 有些图不能被识别
                except:
                    print(image_filename)
                    continue
                grayscale_image = tf.image.rgb_to_grayscale(image)              # 转换为灰度图像
                resized_image = tf.image.resize_images(grayscale_image, size=[250, 251])   # 全部都转成相同大小
                image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()

                image_label = breed.encode("utf-8")
                example = tf.train.Example(feature={
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
                })
                writer.write(example.SerializeToString())
                writer.close()

                write_records_file()

write_records_file(testing_dataset, ".\\output\\testing-images\\testing-image")
write_records_file(training_dataset, ".\\output\\training-images\\training-image")
