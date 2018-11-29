import pandas as pd
import tensorflow as tf
import glob, os, sys
from random import shuffle
from matplotlib import image as im
from PIL import Image
import numpy as np


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_list_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))



def _convert_to_example(image_buffer, label_int):
	example = tf.train.Example(
		features=tf.train.Features(
			feature={
				'label':_int64_feature(label_int), # model expects 1-based
				'img':_bytes_feature(image_buffer)
			}))
	return example


class ImageCoder(object):
	def __init__(self):
		# Create a single Session to run all image coding calls.
		self._sess = tf.Session()

		# Initializes function that decodes RGB JPEG data.
		self._decode_png_data = tf.placeholder(dtype=tf.string)
		self._decode_png = tf.image.decode_png(self._decode_png_data, channels=1)

	def decode_png(self, image_data):
		image = self._sess.run(
			self._decode_png, feed_dict={self._decode_png_data:image_data})
		return image

	def __del__(self):
		self._sess.close()


def _get_image_data(filename, coder):
	# Read the image file.
	with tf.gfile.FastGFile(filename, 'rb') as ifp:
		image_data = ifp.read()

	# Decode the RGB JPEG.
	image = coder.decode_png(image_data)

	# Check that image converted to RGB
	height = image.shape[0]
	width = image.shape[1]

	return image_data, height, width


def convert_to_example(filename, label):

	# coder = ImageCoder()
	# image_buffer, height, width = _get_image_data(filename, coder)
	# del coder

	image_buffer = load_image(filename)

	example = _convert_to_example(image_buffer, label)
	# yield example.SerializeToString()
	return example.SerializeToString()



def load_image(infilename):
	# with tf.gfile.GFile(infilename, 'rb') as fid:
	# 	image = fid.read()
	image = Image.open(infilename)
	image = np.asarray(image, np.uint8)
	return image.tobytes()

	# return image

def createDataRecord(out_filename, addrs, labels):
	writer = tf.python_io.TFRecordWriter(out_filename)
	for i in range(len(addrs)):
		if not i % 1000:
			print('Train data: {}/{}'.format(i, len(addrs)))
			sys.stdout.flush()
		img = load_image(addrs[i])

		if img is None:
			continue

		feature = {
			'img':_bytes_feature(img),
			'label0':_int64_feature(labels[i][0]),
			'label1':_int64_feature(labels[i][1]),
			'label2':_int64_feature(labels[i][2]),
			'label3':_int64_feature(labels[i][3])
		}
		example = tf.train.Example(features=tf.train.Features(feature=feature))

		writer.write(example.SerializeToString())

	writer.close()
	sys.stdout.flush()


df = pd.read_csv('pilotCNNselection.csv', header=0, usecols=['Id', 'Nucleoplasm', 'Nuclear.membrane', 'Nucleoli',
                                                             'pilotCNN'])

image_path = 'train_nucleus/*tif'
addrs = glob.glob(image_path)

labels = []

for i, addr in enumerate(addrs):
	id = os.path.basename(addr).split('_')[0]
	label = df.loc[df['Id'] == id].values[0][1:4]
	label = label * 1
	label = label.tolist()
	if 1 in label:
		label.append(0)
	else:
		label.append(1)

	labels.append(label)

c = list(zip(addrs, labels))
shuffle(c)
addrs, labels = zip(*c)

train_addrs = addrs[0:int(0.5 * len(addrs))]
train_labels = labels[0:int(0.5 * len(labels))]
# val_addrs = addrs[int(0.8*len(addrs)):int(0.9*len(addrs))]
# val_labels = labels[int(0.8*len(addrs)):int(0.9*len(addrs))]
test_addrs = addrs[int(0.9 * len(addrs)):]
test_labels = labels[int(0.9 * len(labels)):]

createDataRecord('train.tfrecord', train_addrs, train_labels)
# createDataRecord('val.tfrecords', val_addrs, val_labels)
createDataRecord('test.tfrecord', test_addrs, test_labels)
