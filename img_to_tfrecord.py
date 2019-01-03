import pandas as pd
import tensorflow as tf
import glob, os, sys
from random import shuffle
from PIL import Image
import numpy as np


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_list_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def load_image(infilename):
	image = Image.open(infilename)
	image = np.asarray(image, np.uint8)
	return image.tobytes()

	# return image

def createDataRecord(out_dir, addrs, labels):

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	writer = tf.python_io.TFRecordWriter(os.path.join(out_dir, 'data.tfrecord'))
	for i in range(len(addrs)):
		if not i % 1000:
			print('create data: {}/{}'.format(i, len(addrs)))
			sys.stdout.flush()
		img = load_image(addrs[i])

		if img is None:
			continue

		feature = {
			'img':_bytes_feature(img),
			'label0':_int64_feature(labels[i][0]),
			'label1':_int64_feature(labels[i][1]),
			'label2':_int64_feature(labels[i][2]),
			'label3':_int64_feature(labels[i][3]),
			'label4':_int64_feature(labels[i][4]),
			'label5':_int64_feature(labels[i][5]),
			'label6':_int64_feature(labels[i][6])
		}
		example = tf.train.Example(features=tf.train.Features(feature=feature))

		writer.write(example.SerializeToString())

	writer.close()
	sys.stdout.flush()


df = pd.read_csv('pilotCNNselection.csv', header=0, usecols=['Id', 'Nucleoplasm', 'Nuclear.membrane', 'Nucleoli',
                                                             'Nucleoli.fibrillar.center', 'Nuclear.speckles','Nuclear.bodies',
															 'pilotCNN'])

image_path = 'train_nucleus/*tif'
addrs = glob.glob(image_path)
labels = []
num_class = 7
j = 0
for i, addr in enumerate(addrs):
	id = os.path.basename(addr).split('_')[0]
	label = df.loc[df['Id'] == id].values[0][1:num_class]
	label = label * 1
	label = label.tolist()
	if 1 in label:
		label.append(0)
	else:
		label.append(1)
		j = j + 1
	labels.append(label)

print(j)

c = list(zip(addrs, labels))
shuffle(c)
addrs, labels = zip(*c)

train_addrs = addrs[0:int(0.8 * len(addrs))]
train_labels = labels[0:int(0.8 * len(labels))]
val_addrs = addrs[int(0.8*len(addrs)):int(0.9*len(addrs))]
val_labels = labels[int(0.8*len(addrs)):int(0.9*len(addrs))]
test_addrs = addrs[int(0.9 * len(addrs)):]
test_labels = labels[int(0.9 * len(labels)):]

cwd = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(cwd, 'tfrecord')

createDataRecord(os.path.join(data_dir,'train'), train_addrs, train_labels)
createDataRecord(os.path.join(data_dir,'val'), val_addrs, val_labels)
createDataRecord(os.path.join(data_dir,'test'), test_addrs, test_labels)
