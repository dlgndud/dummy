import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np

class DataSet(object):

    def __init__(self, images, labels, img_names, cls):
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._img_names = img_names
        self._cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def img_names(self):
        return self._img_names

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    @property
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples #assert 강한 조건임을 명시, 오류시 에러처리
        end  = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def load_train(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []

    print('Going to read training images')
    for fields in classes:
        index = classes.index(fields)
        #print('Now Going to read {} field (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*g')
        #path = os.path.join(train_path, fields) error~~ why???
        #print('path --> ', path)
        files = glob.glob(path)

        for fl in files:
            #print('Load_train 2nd for loop : ',fl, ' index : ', index, ' field : ', fields)
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))  #np.zeros 배열의 사이즈만큼 0을 채워줌 [0, 0]의 값 생성
            label[index] = 1.0 #dog [1,0]. cat[0, 1]
            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls



    print('Going to read training images')
    for fields in classes:
        index = classes.index(fields)
        #print('Now going to read {} files (Index:{})',format(fields, index))
        path = os.path.join(train_path, fields, '*g')

def read_train_sets(train_path, image_size, classes, validation_size):
    class DataSets(object):
        pass #try, catch에서 오류가 있더라도 구문을 진행시킴
    data_sets = DataSets()

    #print('1')
    images, labels, img_names, cls = load_train(train_path, image_size, classes)
    #print('2')
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls) # 각각의 객체를 섞음
    #print('3')
    '''
    for im in img_names:
        print('Image Name ',im)
    '''
    #print('4')
    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])
        #print('before 5 - validation_size, ',validation_size)
    #print('5')
    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_img_names = img_names[:validation_size]
    validation_cls = cls[:validation_size]
    #print('6')
    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_img_names = img_names[validation_size:]
    train_cls = cls[validation_size:]
    #print('7')
    data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
    data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)
    #print('8')
    return data_sets
