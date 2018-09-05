
import os
import numpy as np
import keras.backend as K
import tensorflow as tf
from PIL import Image
from keras.utils import to_categorical
from keras.preprocessing.image import *

from keras.applications.imagenet_utils import preprocess_input

CLASSES = 21
SEGMENTATION_IMAGE_DIR = 'ImageSets/Segmentation/'
TRAIN_LIST_FILE_NAME = 'train.txt'
TRAIN_CLASS_FILE_NAME = 'trainval.txt'
VALIDATION_LIST_FILE_NAME = 'val.txt'

def crossentropy_without_ambiguous(y_true, y_pred):

    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

#    class_weight = [0.2, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 
#                    5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
#    log_softmax = log_softmax * np.array(class_weight)
    

#    y_true = K.one_hot(tf.to_int32(K.flatten(K.argmax(y_true))), K.int_shape(y_pred)[-1]+1)
    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)
    
    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)
    
    return cross_entropy_mean

def categorical_accuracy_without_ambiguous(y_true, y_pred):

#    legal_labels = tf.not_equal(K.argmax(y_true, axis=-1), CLASSES)
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    
    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), nb_classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    return K.sum(tf.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float(legal_labels))


def categorical_accuracy_only_valid_classes(y_true, y_pred):
    
#    legal_labels = tf.not_equal(K.argmax(y_true, axis=-1), CLASSES)
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    
    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), nb_classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    y_true = tf.stack(unpacked[:-1], axis=-1)    
    forground = tf.not_equal(K.argmax(y_true, axis=-1), 0)
    
    return K.sum(tf.to_float(forground & legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float(legal_labels & forground))

def pair_random_crop(x, y, random_crop_size, data_format, sync_seed=None, **kwargs):
    np.random.seed(sync_seed)
    if data_format == 'channels_first':
        h, w = x.shape[1], x.shape[2]
    elif data_format == 'channels_last':
        h, w = x.shape[0], x.shape[1]
    rangeh = (h - random_crop_size[0]) // 2
    rangew = (w - random_crop_size[1]) // 2
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    
    h_start, h_end = offseth, offseth + random_crop_size[0]
    w_start, w_end = offsetw, offsetw + random_crop_size[1]
    if data_format == 'channels_first':
        return x[:, h_start:h_end, w_start:w_end], y[:, h_start:h_end, h_start:h_end]
    elif data_format == 'channels_last':
        return x[h_start:h_end, w_start:w_end, :], y[h_start:h_end, w_start:w_end, :]

def pair_center_crop(x, y, center_crop_size, data_format, **kwargs):
    if data_format == 'channels_first':
        centerh, centerw = x.shape[1] // 2, x.shape[2] // 2
    elif data_format == 'channels_last':
        centerh, centerw = x.shape[0] // 2, x.shape[1] // 2
    lh, lw = center_crop_size[0] // 2, center_crop_size[1] // 2
    rh, rw = center_crop_size[0] - lh, center_crop_size[1] - lw
    
    h_start, h_end = centerh - lh, centerh + rh
    w_start, w_end = centerw - lw, centerw + rw
    if data_format == 'channels_first':
        return x[:, h_start:h_end, w_start:w_end], \
            y[:, h_start:h_end, w_start:w_end]
    elif data_format == 'channels_last':
        return x[h_start:h_end, w_start:w_end, :], \
            y[h_start:h_end, w_start:w_end, :]

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
                 zoom_maintain_shape=True,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 crop_mode='none',
                 crop_size=(0, 0),
                 pad_size=None,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None,
                 validation_split=0.0,
                 label_cval=255):
        if data_format is None:
            data_format = K.image_data_format()
        if data_format not in {'channels_last', 'channels_first'}:
            raise Exception('data_format should be channels_last (channel after row and '
                            'column) or channels_first (channel before row and column). '
                            'Received arg: ', data_format)
        self.data_format = data_format
        self.image_shape = image_shape
        self.rescale = rescale
        if crop_mode not in {'none', 'random', 'center'}:
            raise Exception('crop_mode should be "none" or "random" or "center" '
                            'Received arg: ', crop_mode)
        self.crop_mode = crop_mode
        self.crop_size = crop_size
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.pad_size = pad_size
        self.fill_mode = fill_mode
        self.label_cval = label_cval
        self.horizontal_flip=horizontal_flip,
        self.vertical_flip=vertical_flip,
        self.featurewise_center = featurewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.channel_shift_range = channel_shift_range
        self.rotation_range = rotation_range
        self.preprocessing_function = preprocessing_function
        self.samplewise_center = samplewise_center
        self.samplewise_std_normalization =samplewise_std_normalization
        self.cval = cval
        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise Exception('zoom_range should be a float or '
                            'a tuple or list of two floats. '
                            'Received arg: ', zoom_range)
        self.zoom_maintain_shape = zoom_maintain_shape

        super(VocImageDataGenerator, self).__init__()

    def flow_from_imageset(self, directory,
                        target_size=(256, 256), normalize=False,
                        classes=None, class_mode='categorical',
                        loss_shape=None, ignore_label=255,
                        batch_size=32, shuffle=True, seed=None):
        if self.crop_mode == 'random' or self.crop_mode == 'center':
            target_size = self.crop_size
        return VocImageIterator(
            directory, self,
            target_size=target_size,
            crop_mode=self.crop_mode,
            pad_size=self.pad_size,
            loss_shape=loss_shape, ignore_label=ignore_label,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            label_cval = self.label_cval,
            normalize = normalize,
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
               
    def random_transform(self, x, y):
#        # x is a single image, so it doesn't have image number at index 0
#        img_row_index = 0 if self.data_format == 'channels_last' else 1
#        img_col_index = 1 if self.data_format == 'channels_last' else 2
#        img_channel_index = 2 if self.data_format == 'channels_last' else 0 
#
#        if self.crop_mode == 'none':
#            crop_size = (x.shape[img_row_index], x.shape[img_col_index])
#        else:
#            crop_size = self.crop_size
#        
#        assert x.shape[img_row_index] == y.shape[img_row_index] and x.shape[img_col_index] == y.shape[
#img_col_index], 'DATA ERROR: Different shape of data and label!\ndata shape: %s, label shape: %s' % (str(x.shape), str(y.shape))
#
#        # rotation
#        if self.rotation_range:
#            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
#        else:
#            theta = 0
#        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
#                                    [np.sin(theta), np.cos(theta), 0],
#                                    [0, 0, 1]])
#        if self.height_shift_range:
#            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * crop_size[0]
#        else:
#            tx = 0
#
#        if self.width_shift_range:
#            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * crop_size[1]
#        else:
#            ty = 0
#
#        translation_matrix = np.array([[1, 0, tx],
#                                       [0, 1, ty],
#                                       [0, 0, 1]])
#        if self.shear_range:
#            shear = np.random.uniform(-self.shear_range, self.shear_range)
#        else:
#            shear = 0
#        shear_matrix = np.array([[1, -np.sin(shear), 0],
#                                 [0, np.cos(shear), 0],
#                                 [0, 0, 1]])
#
#        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
#            zx, zy = 1, 1
#        else:
#            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
#        if self.zoom_maintain_shape:
#            zy = zx
#        zoom_matrix = np.array([[zx, 0, 0],
#                                  [0, zy, 0],
#                                  [0, 0, 1]])
#                              
#        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)
#
#        h, w = x.shape[img_row_index], x.shape[img_col_index]
#        transform_matrix = ImageDataGenerator.transform_matrix_offset_center(transform_matrix, h, w)
#
#        x = ImageDataGenerator.apply_transform(x, transform_matrix, img_channel_index, fill_mode=self.fill_mode, cval=self.cval)
#        y = ImageDataGenerator.apply_transform(y, transform_matrix, img_channel_index, fill_mode='constant', cval=self.label_cval)
#        print ("* illegal pixel:{}".format(np.sum(y > 21) - np.sum(y == 255)))
#    
#        if self.channel_shift_range != 0:
#            x = random_channel_shift(x, self.channel_shift_range, img_channel_index)
#        if self.horizontal_flip:
#            if np.random.random() < 0.5:
#                x = flip_axis(x, img_col_index)
#                y = flip_axis(y, img_col_index)
#                      
#        if self.vertical_flip:
#            if np.random.random() < 0.5:
#                x = flip_axis(x, img_row_index)
#                y = flip_axis(y, img_row_index)
               
        params = ImageDataGenerator.get_random_transform(self, img_shape=x.shape, seed=None)
        
        x = ImageDataGenerator.apply_transform(self, x, params)
        fill_mode = self.fill_mode
        self.fill_mode = 'constant'
        y = ImageDataGenerator.apply_transform(self, y, params)
        self.fill_mode = fill_mode
        
        if self.crop_mode == 'center':
            x, y = pair_center_crop(x, y, self.crop_size, self.data_format)
        elif self.crop_mode == 'random':
            x, y = pair_random_crop(x, y, self.crop_size, self.data_format)

#        print ("** illegal pixel:{}".format(np.sum(y > 21) - np.sum(y == 255)))

        return x, y

class VocImageIterator(Iterator):
    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 crop_mode='none', pad_size=None, 
                 ignore_label=255, label_cval=255,
                 batch_size=32, shuffle=False, seed=None,
                 data_format=None, loss_shape=None,
                 normalize=False):
        if data_format is None:
            data_format = backend.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.ignore_label = ignore_label
        self.crop_mode = crop_mode
        self.label_cval = label_cval
        self.data_format = data_format
        self.loss_shape = loss_shape
        self.pad_size = pad_size
        self.normalize = normalize
        
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
#        if class_mode == 'categorical':
#            self.label_shape = target_size + (classes,)
#        elif class_mode == 'binary':
#            self.label_shape = target_size + (1,)
        self.label_shape = target_size + (1,)
        self.class_mode = class_mode
        
        self.train_filenames, self.label_filenames = get_train_files(directory)
        
        self.samples = len(self.train_filenames)

        super(VocImageIterator, self).__init__(self.samples,
                                 batch_size,
                                 shuffle,
                                 seed)

    def _get_batches_of_transformed_samples(self, index_array):

        current_batch_size = len(index_array)
        if self.target_size:
            batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
            batch_y = np.zeros((current_batch_size,) + self.label_shape, dtype=np.int8)

        grayscale = self.color_mode == 'grayscale'
        
        for i, j in enumerate(index_array):
            img = load_img(self.train_filenames[j], grayscale, target_size=None)
            label = Image.open(self.label_filenames[j])

            if self.target_size is not None:
                if self.crop_mode != 'none':
                    x = img_to_array(img, data_format=self.data_format)
                    y = img_to_array(label, data_format=self.data_format).astype(int)
 
                    img_w, img_h = img.size
                    if self.pad_size:
                        pad_w = max(self.pad_size[1] - img_w, 0)
                        pad_h = max(self.pad_size[0] - img_h, 0)
                    else:
                        pad_w = max(self.target_size[1] - img_w, 0)
                        pad_h = max(self.target_size[0] - img_h, 0)
                    if self.data_format == 'channels_first':
                        x = np.lib.pad(x, ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)), 'constant', constant_values=0.)
                        y = np.lib.pad(y, ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
                                       'constant', constant_values=self.label_cval)
                    elif self.data_format == 'channels_last':
                        x = np.lib.pad(x, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)), 'constant', constant_values=0.)
                        y = np.lib.pad(y, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)), 'constant', constant_values=self.label_cval)
                else:
                    x = img_to_array(img.resize((self.target_size[1], self.target_size[0]),
                            Image.BILINEAR), data_format=self.data_format)
                    y = img_to_array(label.resize((self.target_size[1], self.target_size[0]), 
                            Image.NEAREST), data_format=self.data_format).astype(int)

            else:
                batch_x = np.zeros((current_batch_size,) + x.shape)
                if self.loss_shape is not None:
                    batch_y = np.zeros((current_batch_size,) + self.loss_shape)
                else:
                    batch_y = np.zeros((current_batch_size,) + self.label_shape)
            
            if self.normalize:
                x = x / 255.
            x, y = self.image_data_generator.random_transform(x, y)
            x = self.image_data_generator.standardize(x)
            
#            print ("illegal pixel:{}".format(np.sum(y > 21) - np.sum(y == 255)))

            if self.ignore_label:
                y[np.where(y == self.ignore_label)] = self.classes
    
#            print ("y.shape:{}".format(y.shape))
            if self.loss_shape is not None:
                y = np.reshape(y, self.loss_shape)
#            max_val = np.max(y)
#            print ("* y.shape:{}".format(y.shape))

#            y = to_categorical(y, self.classes + 1)

            batch_x[i] = x
            batch_y[i] = y
                
        batch_x = preprocess_input(batch_x)
        
        if self.class_mode == 'binary':
            return batch_x

#        print ("batch_y.shape:{}".format(batch_y.shape))
        return batch_x, batch_y
               
#    def next(self):
#        with self.lock:
#            index_array = next(self.image_data_generator)
#        return _get_batches_of_transformed_samples(index_array)

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


