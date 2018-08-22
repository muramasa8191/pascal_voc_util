
import os
import numpy as np
import keras.backend as K
from PIL import Image
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, img_to_array, Iterator, load_img

CLASSES = 21
SEGMENTATION_IMAGE_DIR = 'ImageSets/Segmentation/'
TRAIN_LIST_FILE_NAME = 'train.txt'
TRAIN_CLASS_FILE_NAME = 'trainval.txt'
VALIDATION_LIST_FILE_NAME = 'val.txt'

class VocImageDataGenerator(ImageDataGenerator):
    def __init__(self,
                 image_shape=(224, 224, 3),
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None,
                 validation_split=0.0):
        self.image_shape = image_shape
        super(VocImageDataGenerator, self).__init__(featurewise_center=featurewise_center,
                                                    samplewise_center=samplewise_center,
                                                    featurewise_std_normalization=featurewise_std_normalization,
                                                    samplewise_std_normalization=samplewise_std_normalization,
                                                    zca_whitening=zca_whitening,
                                                    zca_epsilon=zca_epsilon,
                                                    rotation_range=rotation_range,
                                                    width_shift_range=width_shift_range,
                                                    height_shift_range=height_shift_range,
                                                    brightness_range=brightness_range,
                                                    shear_range=shear_range,
                                                    zoom_range=zoom_range,
                                                    channel_shift_range=channel_shift_range,
                                                    fill_mode=fill_mode,
                                                    cval=cval,
                                                    horizontal_flip=horizontal_flip,
                                                    vertical_flip=vertical_flip,
                                                    rescale=rescale,
                                                    preprocessing_function=preprocessing_function,
                                                    data_format=data_format,
                                                    validation_split=validation_split)

    def flow_from_imageset(self, directory,
                        target_size=(256, 256),
                        classes=None, class_mode='categorical',
                        batch_size=32, shuffle=True, seed=None):
        return VocImageIterator(
            directory, self,
            target_size=target_size,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed)

    def standardize(self, x):
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.samplewise_center:
            x -= np.mean(x, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, keepdims=True) + K.epsilon())

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This ImageDataGenerator specifies '
                '`featurewise_center`, but it hasn\'t '
                'been fit on any training data. Fit it '
                'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + K.epsilon())
            else:
                warnings.warn('This ImageDataGenerator specifies '
                          '`featurewise_std_normalization`, '
                          'but it hasn\'t '
                          'been fit on any training data. Fit it '
                          'first by calling `.fit(numpy_data)`.')
        return x

    def fit(self, x):

        if self.featurewise_center:
            self.mean = np.mean(x, axis=0)

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=0)

#        print ("called standardize: mean={}, std={}".format(self.mean, self.std))
               
class VocImageIterator(Iterator):
    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=False, seed=None,
                 data_format=None):
        if data_format is None:
            data_format = backend.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.data_format = data_format
        channel = 3
        if color_mode != 'rgb':
            channel = 1
        self.color_mode = color_mode
        if self.data_format == 'channels_last':
            self.image_shape = self.target_size + (channel,)
        else:
            self.image_shape = (channel,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse', 'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                            '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        if class_mode == 'categorical':
            self.label_shape = target_size + (classes,)
        elif class_mode == 'binary':
            self.label_shape = target_size + (1,)

        self.train_filenames, self.label_filenames = get_train_files(directory)
        if self.image_data_generator.featurewise_center or self.image_data_generator.featurewise_std_normalization: 
            images = []
            for filename in self.train_filenames:
                image = load_img(filename, False, self.image_shape)
                image = img_to_array(image, data_format=self.data_format)
                images.append(image)
                
            self.image_data_generator.fit(images)
        
        self.samples = len(self.train_filenames)

        super(VocImageIterator, self).__init__(self.samples,
                                 batch_size,
                                 shuffle,
                                 seed)

    def _get_batches_of_transformed_samples(self, index_array):
#        with self.lock:
#            index_array = next(self.index_generator)
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        batch_y = np.zeros((len(index_array),) + self.label_shape, dtype=np.int8)

        grayscale = self.color_mode == 'grayscale'
        
        for i, j in enumerate(index_array):
            x = load_img(self.train_filenames[j], grayscale, self.image_shape)
            x = img_to_array(x, data_format=self.data_format)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            y = Image.open(self.label_filenames[j])
            if y.size != self.target_size:
                y = y.resize(self.target_size)
            y = img_to_array(y, self.data_format)
            y[y == 255] = 0
            y = to_categorical(y, self.classes)
            batch_y[i] = y
        
        return batch_x, batch_y
               
def image_generator(file_paths, size=None, normalization=True):
    """ generate train data and val
    Parameters
    ----------
    file_paths: the arrray of the path to get the image
    size: input size for model. the image will be resized by this size
    normalization: if True, each pixel value will be devided by 255.
    """
    for file_path in file_paths:
        if file_path.endswith(".png") or file_path.endswith(".jpg"):
            # open a image
            image = Image.open(file_path)
            # resize by init_size
            if size is not None and size != image.size:
                image = image.resize(size)
            # delete alpha channel
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image = np.asarray(image)
#            if normalization:
##                image = image / 255.0
#                image 
            yield image

def pascal_data_generator(data_paths, val_paths, size=None):
    """ generate data array
    Parameters
    ----------
    train_data_paths: array of the paths for train data
    train_val_paths: array of the paths for train value
    size: size of the image
    Returns
    --------
    
    """
    img_org, img_segmented = [], []
    for image in image_generator(data_paths, size):
        img_org.append(image)
    
    for image in image_generator(val_paths, size, normalization=False):
        img_segmented.append(image)
    
    assert len(img_org) == len(img_segmented)
    
    # Cast to nparray
    img_data = np.asarray(img_org, dtype=np.float32)
    img_segmented = np.asarray(img_segmented, dtype=np.uint8)

    # Cast void pixel to bg
    img_segmented = np.where(img_segmented == 255, 0, img_segmented)

    # change 0 - 21
#    identity = np.identity(CLASSES, dtype=np.uint8)
#    img_segmented = identity[img_segmented]
    img_segmented = to_categorical(img_segmented, CLASSES)

    return [img_data, img_segmented] 

def get_train_files(root_dir):
    
    train_data_files = []
    train_class_files = []
    
    path = os.path.join(root_dir, SEGMENTATION_IMAGE_DIR)
    
    img_dir = os.path.join(root_dir + '/', 'JPEGImages/')
    seg_dir = os.path.join(root_dir + '/', 'SegmentationClass/')
    
    with open(os.path.join(path, TRAIN_LIST_FILE_NAME)) as f:
        for s in f:
            file_name = s.rstrip('\r\n')
            train_data_files.append(os.path.join(img_dir, file_name + '.jpg'))
            train_class_files.append(os.path.join(seg_dir, file_name + '.png'))

    return [train_data_files, train_class_files]

def get_val_files(root_dir):
    
    val_data_files = []
    val_class_files = []
    
    path = os.path.join(root_dir, SEGMENTATION_IMAGE_DIR)
    
    img_dir = os.path.join(root_dir + '/', 'JPEGImages/')
    seg_dir = os.path.join(root_dir + '/', 'SegmentationClass/')
    
    with open(os.path.join(path, VALIDATION_LIST_FILE_NAME)) as f:
        for s in f:
            file_name = s.rstrip('\r\n')
            val_data_files.append(os.path.join(img_dir, file_name + '.jpg'))
            val_class_files.append(os.path.join(seg_dir, file_name + '.png'))
    
    return [val_data_files, val_class_files]

def get_class_map():
    """Return the text of each class
        0 - background
        1 to 20 - classes
        255 - void region
        Returns
        -------
        classes_dict : dict
    """
    
    class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'potted-plant',
                   'sheep', 'sofa', 'train', 'tv/monitor', 'ambigious']
                   
    # dict for class names except for void
    classes_dict = list(enumerate(class_names[:-1]))
    # add void
    classes_dict.append((255, class_names[-1]))
                   
    classes_dict = dict(classes_lut)

    return classes_dict


