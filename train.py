import os

from absl import app as absl_app
from absl import flags

import tensorflow as tf

from utils.flags import core as flags_core
from utils.logs import logger
import resnet_model
import resnet_run_loop

_HEIGHT = 512
_WIDTH = 512
_NUM_CHANNELS = 1
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES
_NUM_CLASSES = 4
_NUM_DATA_FILES = 1



_NUM_IMAGES = {
	'train':3107, # total train 3107  # test: 6215
	'validation':3107,
}

DATASET_NAME = 'PROTEIN'


###############################################################################
# Data processing
###############################################################################


def parse_record(raw_record, is_training, dtype):

	keys_to_features = {
		"img":tf.FixedLenFeature([], tf.string),
		'label0':tf.FixedLenFeature([], tf.int64),
		'label1':tf.FixedLenFeature([], tf.int64),
		'label2':tf.FixedLenFeature([], tf.int64),
		'label3':tf.FixedLenFeature([], tf.int64)
	}

	parsed = tf.parse_single_example(raw_record, keys_to_features)
	image = tf.decode_raw(parsed["img"], tf.uint8)
	image = tf.reshape(image, [_HEIGHT, _WIDTH, _NUM_CHANNELS])
	image = tf.cast(tf.transpose(image, [1, 2, 0]), tf.float32)
	image = tf.cast(image, dtype)

	label0 = parsed["label0"]
	label1 = parsed["label1"]
	label2 = parsed["label2"]
	label3 = parsed["label3"]
	label = tf.stack([label0,label1,label2,label3])
	label = tf.cast(label, tf.int32)

	return image, label


def preprocess_image(image, is_training):
	if is_training:
		pass
	return image


def input_fn(is_training, data_dir, batch_size, num_epochs=1,
             dtype=tf.float32, datasets_num_private_threads=None,
             num_parallel_batches=1):
	"""Input function which provides batches for data_batch_train or eval.
	Args:
	  is_training: A boolean denoting whether the input is for training.
	  data_dir: The directory containing the input data.
	  batch_size: The number of samples per batch.
	  num_epochs: The number of epochs to repeat the dataset.
	  dtype: Data type to use for images/features
	  datasets_num_private_threads: Number of private threads for tf.data.
	  num_parallel_batches: Number of parallel batches for tf.data.
	Returns:
	  A dataset that can be used for iteration.
	"""
	filenames = os.path.join(data_dir, 'data')

	# dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES) # this only works for CIFAR
	dataset = tf.data.TFRecordDataset(filenames, buffer_size=_RECORD_BYTES)

	return resnet_run_loop.process_record_dataset(
		dataset=dataset,
		is_training=is_training,
		batch_size=batch_size,
		shuffle_buffer=_NUM_IMAGES['train'],
		parse_record_fn=parse_record,
		num_epochs=num_epochs,
		dtype=dtype,
		datasets_num_private_threads=datasets_num_private_threads,
		num_parallel_batches=num_parallel_batches
	)


def get_synth_input_fn(dtype):
	return resnet_run_loop.get_synth_input_fn(
		_HEIGHT, _WIDTH, _NUM_CHANNELS, _NUM_CLASSES, dtype=dtype)


###############################################################################
# Running the model
###############################################################################
class pilotCNNModel(resnet_model.Model):

	def __init__(self, resnet_size, data_format=None, num_classes=_NUM_CLASSES,
	             resnet_version=resnet_model.DEFAULT_VERSION,
	             dtype=resnet_model.DEFAULT_DTYPE):
		"""These are the parameters that work for dataset.
		Args:
		  resnet_size: The number of convolutional layers needed in the model.
		  data_format: Either 'channels_first' or 'channels_last', specifying which
			data format to use when setting up the model.
		  num_classes: The number of output classes needed from the model. This
			enables users to extend the same model to their own datasets.
		  resnet_version: Integer representing which version of the ResNet network
		  to use. See README for details. Valid values: [1, 2]
		  dtype: The TensorFlow dtype to use for calculations.
		Raises:
		  ValueError: if invalid resnet_size is chosen
		"""
		if resnet_size % 6 != 2:
			raise ValueError('resnet_size must be 6n + 2:', resnet_size)

		num_blocks = (resnet_size - 2) // 6

		super(pilotCNNModel, self).__init__(
			resnet_size=resnet_size,
			bottleneck=False,
			num_classes=num_classes,
			num_filters=8,
			kernel_size=3,
			conv_stride=1,
			first_pool_size=None,
			first_pool_stride=None,
			block_sizes=[num_blocks] * 3,
			block_strides=[1, 2, 2],
			resnet_version=resnet_version,
			data_format=data_format,
			dtype=dtype
		)


def pilotCNN_model_fn(features, labels, mode, params):
	features = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])
	learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
		batch_size=params['batch_size'], batch_denom=64,
		num_images=_NUM_IMAGES['train'], boundary_epochs=[91, 136, 182],
		decay_rates=[1, 0.1, 0.01, 0.001])

	weight_decay = 2e-4

	def loss_filter_fn(_):
		return True

	return resnet_run_loop.resnet_model_fn(
		features=features,
		labels=labels,
		mode=mode,
		model_class=pilotCNNModel,
		resnet_size=params['resnet_size'],
		weight_decay=weight_decay,
		learning_rate_fn=learning_rate_fn,
		momentum=0.9,
		data_format=params['data_format'],
		resnet_version=params['resnet_version'],
		loss_scale=params['loss_scale'],
		loss_filter_fn=loss_filter_fn,
		dtype=params['dtype'],
		fine_tune=params['fine_tune']
	)


def define_flags():
	resnet_run_loop.define_resnet_flags()
	flags.adopt_module_key_flags(resnet_run_loop)
	cwd = os.path.dirname(os.path.realpath(__file__))

	flags_core.set_defaults(data_dir=os.path.join(cwd,'tfrecord'),
	                        model_dir='model/',
	                        resnet_size='14',
	                        train_epochs=1000,
	                        epochs_between_evals=1,
	                        batch_size=3,
	                        image_bytes_as_serving_input=False,
	                        eval_only = False)


def run_pilotCNN(flags_obj):
	"""
	Args:
	  flags_obj: An object containing parsed flag values.
	"""
	if flags_obj.image_bytes_as_serving_input:
		tf.logging.fatal('--image_bytes_as_serving_input cannot be set to True '
		                 'for CIFAR. This flag is only applicable to ImageNet.')
		return

	input_function = (flags_obj.use_synthetic_data and
	                  get_synth_input_fn(flags_core.get_tf_dtype(flags_obj)) or
	                  input_fn)
	resnet_run_loop.resnet_main(
		flags_obj, pilotCNN_model_fn, input_function, DATASET_NAME,
		shape=[_HEIGHT, _WIDTH, _NUM_CHANNELS])


def main(_):
	with logger.benchmark_context(flags.FLAGS):
		run_pilotCNN(flags.FLAGS)


if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	define_flags()
	absl_app.run(main)
