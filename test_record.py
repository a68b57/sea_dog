import tensorflow as tf
import numpy as np
from PIL import Image
import os


_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS
# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1



def get_filenames(is_training, data_dir):
  """Returns a list of filenames."""
  data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

  assert os.path.exists(data_dir), (
      'Run cifar10_download_and_extract.py first to download and extract the '
      'CIFAR-10 data.')

  if is_training:
    return [
        os.path.join(data_dir, 'data_batch_%d.bin' % i)
        for i in range(1, 5 + 1)
    ]
  else:
    return [os.path.join(data_dir, 'test_batch.bin')]


def extract(record):
	features = {
	'img':tf.FixedLenFeature([], tf.string),
	'label0':tf.FixedLenFeature([], tf.int64),
	'label1':tf.FixedLenFeature([], tf.int64),
	'label2':tf.FixedLenFeature([], tf.int64),
	'label3':tf.FixedLenFeature([], tf.int64)
	}

	parsed = tf.parse_single_example(record, features)

	image = tf.decode_raw(parsed["img"], tf.uint8)
	image = tf.reshape(image, [1, 512, 512])
	image = tf.cast(tf.transpose(image, [1, 2, 0]), tf.float32)
	image = tf.cast(image, tf.float32)

	label0 = parsed["label0"]
	label1 = parsed["label1"]
	label2 = parsed["label2"]
	label3 = parsed["label3"]

	label = tf.stack([label0,label1,label2,label3])
	# label = tf.reshape(label, [4,])
	label = tf.cast(label, tf.int32)
	#
	# record_vector = tf.decode_raw(record, tf.uint8)
	# label = tf.cast(record_vector[0], tf.int32)
	# depth_major = tf.reshape(record_vector[1:_RECORD_BYTES],
	#                          [_NUM_CHANNELS, _HEIGHT, _WIDTH])
	# image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
	# image = tf.cast(image, tf.float32)

	return label

filenames = get_filenames(True, 'cifar10_data/')
# dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)
dataset = tf.data.TFRecordDataset('train.tfrecord')

dataset = dataset.prefetch(buffer_size=1)
dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
          extract,
          batch_size=32))

dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

# dataset = tf.data.TFRecordDataset(['data_batch_1.bin'], 512*512)
# dataset = dataset.map(extract)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()


with tf.Session() as sess:
	while True:
		img = sess.run(next_element)
		print(img.shape)
