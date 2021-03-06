
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import multiprocessing
import os

from absl import flags
import tensorflow as tf
from tensorflow.contrib.data.python.ops import threadpool
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops

import resnet_model
from utils.flags import core as flags_core
from utils.export import export
from utils.logs import hooks_helper
from utils.logs import logger
from utils import imagenet_preprocessing
from utils.misc import distribution_utils
from utils.misc import model_helpers

import numpy as np


###############################################################################
# for computing f1-score
###############################################################################

def metric_variable(shape, dtype, validate_shape=True, name=None):
	"""Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES`) collections.
	from https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/metrics_impl.py
	"""
	return variable_scope.variable(
		lambda: array_ops.zeros(shape, dtype),
		trainable=False,
		collections=[ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.METRIC_VARIABLES],
		validate_shape=validate_shape,
		name=name,
	)


def streaming_counts(y_true, y_pred, num_classes):
	y_true = tf.cast(y_true, tf.int64)
	y_pred = tf.cast(y_pred, tf.int64)
	"""Computes the TP, FP and FN counts for the micro and macro f1 scores.
	The weighted f1 score can be inferred from the macro f1 score provided
	we compute the weights also.
	This function also defines the update ops to these counts

	Args:
		y_true (Tensor): 2D Tensor representing the target labels
		y_pred (Tensor): 2D Tensor representing the predicted labels
		num_classes (int): number of possible classes
	Returns:
		tuple: the first element in the tuple is itself a tuple grouping the counts,
		the second element is the grouped update op.
	"""

	# Weights for the weighted f1 score
	weights = metric_variable(
		shape=[num_classes], dtype=tf.int64, validate_shape=False, name="weights"
	)
	# Counts for the macro f1 score
	tp_mac = metric_variable(
		shape=[num_classes], dtype=tf.int64, validate_shape=False, name="tp_mac"
	)
	fp_mac = metric_variable(
		shape=[num_classes], dtype=tf.int64, validate_shape=False, name="fp_mac"
	)
	fn_mac = metric_variable(
		shape=[num_classes], dtype=tf.int64, validate_shape=False, name="fn_mac"
	)
	# Counts for the micro f1 score
	tp_mic = metric_variable(
		shape=[], dtype=tf.int64, validate_shape=False, name="tp_mic"
	)
	fp_mic = metric_variable(
		shape=[], dtype=tf.int64, validate_shape=False, name="fp_mic"
	)
	fn_mic = metric_variable(
		shape=[], dtype=tf.int64, validate_shape=False, name="fn_mic"
	)

	# Update ops, as in the previous section:
	#   - Update ops for the macro f1 score
	up_tp_mac = tf.assign_add(tp_mac, tf.count_nonzero(y_pred * y_true, axis=0))
	up_fp_mac = tf.assign_add(fp_mac, tf.count_nonzero(y_pred * (y_true - 1), axis=0))
	up_fn_mac = tf.assign_add(fn_mac, tf.count_nonzero((y_pred - 1) * y_true, axis=0))

	#   - Update ops for the micro f1 score
	up_tp_mic = tf.assign_add(tp_mic, tf.count_nonzero(y_pred * y_true, axis=None))
	up_fp_mic = tf.assign_add(
		fp_mic, tf.count_nonzero(y_pred * (y_true - 1), axis=None)
	)
	up_fn_mic = tf.assign_add(
		fn_mic, tf.count_nonzero((y_pred - 1) * y_true, axis=None)
	)
	# Update op for the weights, just summing
	up_weights = tf.assign_add(weights, tf.reduce_sum(y_true, axis=0))

	# Grouping values
	counts = (tp_mac, fp_mac, fn_mac, tp_mic, fp_mic, fn_mic, weights)
	updates = tf.group(
		up_tp_mic, up_fp_mic, up_fn_mic, up_tp_mac, up_fp_mac, up_fn_mac, up_weights
	)

	return counts, updates


def streaming_f1(counts):
	"""Computes the f1 scores from the TP, FP and FN counts

	Args:
		counts (tuple): macro and micro counts, and weights in the end

	Returns:
		tuple(Tensor): The 3 tensors representing the micro, macro and weighted
			f1 score
	"""
	# unpacking values
	tp_mac, fp_mac, fn_mac, tp_mic, fp_mic, fn_mic, weights = counts

	# normalize weights
	weights = weights / tf.reduce_sum(weights)

	# computing the micro f1 score
	prec_mic = tp_mic / (tp_mic + fp_mic)
	rec_mic = tp_mic / (tp_mic + fn_mic)
	f1_mic = 2 * prec_mic * rec_mic / (prec_mic + rec_mic)
	f1_mic = tf.reduce_mean(f1_mic)

	# computing the macro and weighted f1 score
	prec_mac = tp_mac / (tp_mac + fp_mac)
	rec_mac = tp_mac / (tp_mac + fn_mac)
	f1_mac = 2 * prec_mac * rec_mac / (prec_mac + rec_mac)
	f1_wei = tf.reduce_sum(f1_mac * weights)
	f1_mac_mean = tf.reduce_mean(f1_mac)

	return f1_mic, f1_mac_mean, f1_wei, f1_mac


def tf_f1_score(y_true, y_pred):
    """Computes 3 different f1 scores, micro macro
    weighted.
    micro: f1 score accross the classes, as 1
    macro: mean of f1 scores per class
    weighted: weighted average of f1 scores per class,
              weighted from the support of each class
    Args:
        y_true (Tensor): labels, with shape (batch, num_classes)
        y_pred (Tensor): model's predictions, same shape as y_true
    Returns:
        tupe(Tensor): (micro, macro, weighted)
                      tuple of the computed f1 scores
    """

    f1s = [0, 0, 0]

    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)

    for i, axis in enumerate([None, 0]):
        TP = tf.count_nonzero(y_pred * y_true, axis=axis)
        FP = tf.count_nonzero(y_pred * (y_true - 1), axis=axis)
        FN = tf.count_nonzero((y_pred - 1) * y_true, axis=axis)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        f1s[i] = tf.reduce_mean(f1)

    weights = tf.reduce_sum(y_true, axis=0)
    weights /= tf.reduce_sum(weights)

    f1s[2] = tf.reduce_sum(f1 * weights)

    micro, macro, weighted = f1s
    return micro, macro, weighted


################################################################################
# Functions for input processing.
################################################################################
def process_record_dataset(dataset,
						   is_training,
						   batch_size,
						   shuffle_buffer,
						   parse_record_fn,
						   num_epochs=1,
						   dtype=tf.float32,
						   datasets_num_private_threads=None,
						   num_parallel_batches=1):
	"""Given a Dataset with raw records, return an iterator over the records.

  Args:
	dataset: A Dataset representing raw records
	is_training: A boolean denoting whether the input is for training.
	batch_size: The number of samples per batch.
	shuffle_buffer: The buffer size to use when shuffling records. A larger
	  value results in better randomness, but smaller values reduce startup
	  time and use less memory.
	parse_record_fn: A function that takes a raw record and returns the
	  corresponding (image, label) pair.
	num_epochs: The number of epochs to repeat the dataset.
	dtype: Data type to use for images/features.
	datasets_num_private_threads: Number of threads for a private
	  threadpool created for all datasets computation.
	num_parallel_batches: Number of parallel batches for tf.data.

  Returns:
	Dataset of (image, label) pairs ready for iteration.
  """

	# Prefetches a batch at a time to smooth out the time taken to load input
	# files for shuffling and processing.
	dataset = dataset.prefetch(buffer_size=batch_size)
	if is_training:
		# Shuffles records before repeating to respect epoch boundaries.
		dataset = dataset.shuffle(buffer_size=shuffle_buffer)

	# Repeats the dataset for the number of epochs to data_batch_1.bin.
	dataset = dataset.repeat(num_epochs)

	# Parses the raw records into images and labels.
	dataset = dataset.apply(
		tf.data.experimental.map_and_batch(
			lambda value:parse_record_fn(value, is_training, dtype),
			batch_size=batch_size,
			num_parallel_batches=num_parallel_batches,
			drop_remainder=False))

	# Operations between the final prefetch and the get_next call to the iterator
	# will happen synchronously during run time. We prefetch here again to
	# background all of the above processing work and keep it out of the
	# critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
	# allows DistributionStrategies to adjust how many batches to fetch based
	# on how many devices are present.
	dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

	# Defines a specific size thread pool for tf.data operations.
	if datasets_num_private_threads:
		tf.logging.info('datasets_num_private_threads: %s',
						datasets_num_private_threads)
		dataset = threadpool.override_threadpool(
			dataset,
			threadpool.PrivateThreadPool(
				datasets_num_private_threads,
				display_name='input_pipeline_thread_pool'))

	return dataset



def image_bytes_serving_input_fn(image_shape, dtype=tf.float32):
	"""Serving input fn for raw jpeg images."""

	def _preprocess_image(image_bytes):
		"""Preprocess a single raw image."""
		# Bounding box around the whole image.
		bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=dtype, shape=[1, 1, 4])
		height, width, num_channels = image_shape
		image = imagenet_preprocessing.preprocess_image(
			image_bytes, bbox, height, width, num_channels, is_training=False)
		return image

	image_bytes_list = tf.placeholder(
		shape=[None], dtype=tf.string, name='input_tensor')
	images = tf.map_fn(
		_preprocess_image, image_bytes_list, back_prop=False, dtype=dtype)
	return tf.estimator.export.TensorServingInputReceiver(
		images, {'image_bytes':image_bytes_list})


def override_flags_and_set_envars_for_gpu_thread_pool(flags_obj):
	"""Override flags and set env_vars for performance.

  These settings exist to test the difference between using stock settings
  and manual tuning. It also shows some of the ENV_VARS that can be tweaked to
  squeeze a few extra examples per second.  These settings are defaulted to the
  current platform of interest, which changes over time.

  On systems with small numbers of cpu cores, e.g. under 8 logical cores,
  setting up a gpu thread pool with `tf_gpu_thread_mode=gpu_private` may perform
  poorly.

  Args:
	flags_obj: Current flags, which will be adjusted possibly overriding
	what has been set by the user on the command-line.
  """
	cpu_count = multiprocessing.cpu_count()
	tf.logging.info('Logical CPU cores: %s', cpu_count)

	# Sets up thread pool for each GPU for op scheduling.
	per_gpu_thread_count = 1
	total_gpu_thread_count = per_gpu_thread_count * flags_obj.num_gpus
	os.environ['TF_GPU_THREAD_MODE'] = flags_obj.tf_gpu_thread_mode
	os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)
	tf.logging.info('TF_GPU_THREAD_COUNT: %s', os.environ['TF_GPU_THREAD_COUNT'])
	tf.logging.info('TF_GPU_THREAD_MODE: %s', os.environ['TF_GPU_THREAD_MODE'])

	# Reduces general thread pool by number of threads used for GPU pool.
	main_thread_count = cpu_count - total_gpu_thread_count
	flags_obj.inter_op_parallelism_threads = main_thread_count

	# Sets thread count for tf.data. Logical cores minus threads assign to the
	# private GPU pool along with 2 thread per GPU for event monitoring and
	# sending / receiving tensors.
	num_monitoring_threads = 2 * flags_obj.num_gpus
	flags_obj.datasets_num_private_threads = (cpu_count - total_gpu_thread_count
											  - num_monitoring_threads)


################################################################################
# Functions for running training/eval/validation loops for the model.
################################################################################
def learning_rate_with_decay(
		batch_size, batch_denom, num_images, boundary_epochs, decay_rates,
		base_lr=0.1, warmup=False):
	"""Get a learning rate that decays step-wise as training progresses.

  Args:
	batch_size: the number of examples processed in each training batch.
	batch_denom: this value will be used to scale the base learning rate.
	  `0.1 * batch size` is divided by this number, such that when
	  batch_denom == batch_size, the initial learning rate will be 0.1.
	num_images: total number of images that will be used for training.
	boundary_epochs: list of ints representing the epochs at which we
	  decay the learning rate.
	decay_rates: list of floats representing the decay rates to be used
	  for scaling the learning rate. It should have one more element
	  than `boundary_epochs`, and all elements should have the same type.
	base_lr: Initial learning rate scaled based on batch_denom.
	warmup: Run a 5 epoch warmup to the initial lr.
  Returns:
	Returns a function that takes a single argument - the number of batches
	trained so far (global_step)- and returns the learning rate to be used
	for training the next batch.
  """
	initial_learning_rate = base_lr * batch_size / batch_denom
	batches_per_epoch = num_images / batch_size

	# Reduce the learning rate at certain epochs.
	# CIFAR-10: divide by 10 at epoch 100, 150, and 200
	# ImageNet: divide by 10 at epoch 30, 60, 80, and 90
	boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
	vals = [initial_learning_rate * decay for decay in decay_rates]

	def learning_rate_fn(global_step):
		"""Builds scaled learning rate function with 5 epoch warm up."""
		lr = tf.train.piecewise_constant(global_step, boundaries, vals)
		if warmup:
			warmup_steps = int(batches_per_epoch * 5)
			warmup_lr = (
				initial_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(
					warmup_steps, tf.float32))
			return tf.cond(global_step < warmup_steps, lambda:warmup_lr, lambda:lr)
		return lr

	return learning_rate_fn


def resnet_model_fn(features, labels, mode, model_class,
					resnet_size, weight_decay, learning_rate_fn, momentum,
					data_format, resnet_version, loss_scale,
					loss_filter_fn=None, dtype=resnet_model.DEFAULT_DTYPE,
					fine_tune=False):
	"""Shared functionality for different resnet model_fns.

  Initializes the ResnetModel representing the model layers
  and uses that model to build the necessary EstimatorSpecs for
  the `mode` in question. For training, this means building losses,
  the optimizer, and the data_batch_1.bin op that get passed into the EstimatorSpec.
  For evaluation and prediction, the EstimatorSpec is returned without
  a data_batch_1.bin op, but with the necessary parameters for the given mode.

  Args:
	features: tensor representing input images
	labels: tensor representing class labels for all input images
	mode: current estimator mode; should be one of
	  `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
	model_class: a class representing a TensorFlow model that has a __call__
	  function. We assume here that this is a subclass of ResnetModel.
	resnet_size: A single integer for the size of the ResNet model.
	weight_decay: weight decay loss rate used to regularize learned variables.
	learning_rate_fn: function that returns the current learning rate given
	  the current global_step
	momentum: momentum term used for optimization
	data_format: Input format ('channels_last', 'channels_first', or None).
	  If set to None, the format is dependent on whether a GPU is available.
	resnet_version: Integer representing which version of the ResNet network to
	  use. See README for details. Valid values: [1, 2]
	loss_scale: The factor to scale the loss for numerical stability. A detailed
	  summary is present in the arg parser help text.
	loss_filter_fn: function that takes a string variable name and returns
	  True if the var should be included in loss calculation, and False
	  otherwise. If None, batch_normalization variables will be excluded
	  from the loss.
	dtype: the TensorFlow dtype to use for calculations.
	fine_tune: If True only data_batch_1.bin the dense layers(final layers).

  Returns:
	EstimatorSpec parameterized according to the input params and the
	current mode.
  """

	# Generate a summary node for the images
	tf.summary.image('images', features['x'], max_outputs=8)
	# Checks that features/images have same data type being used for calculations.
	assert features['x'].dtype == dtype

	model = model_class(resnet_size, data_format, resnet_version=resnet_version,
						dtype=dtype)


	logits = model(features['x'], mode == tf.estimator.ModeKeys.TRAIN)

	# This acts as a no-op if the logits are already in fp32 (provided logits are
	# not a SparseTensor). If dtype is is low precision, logits must be cast to
	# fp32 for numerical stability.
	logits = tf.cast(logits, tf.float32)

	def multi_label_hot(prediction, threshold=0.3):
		prediction = tf.cast(prediction, tf.float32)
		threshold = float(threshold)
		return tf.cast(tf.greater(prediction, threshold), tf.int64)


	predictions = {
		'k_hot_predictions': multi_label_hot(tf.nn.sigmoid(logits)),
		'probabilities': tf.nn.sigmoid(logits, name='sigmoid_tensor'),
		'logits': logits,
		'labels': features['y']
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		# Return the predictions and the specification for serving a SavedModel
		return tf.estimator.EstimatorSpec(
			mode=mode,
			predictions=predictions,
			export_outputs={
				'predict':tf.estimator.export.PredictOutput(predictions)
			})

	# Calculate loss, which includes softmax cross entropy and L2 regularization.
	# cross_entropy = tf.losses.sparse_softmax_cross_entropy(
	#     logits=logits, labels=labels)

	# cross_entropy = tf.losses.softmax_cross_entropy(
		#     logits=logits, one_labels=labels)


	# tf.losses.sigmoid_cross_entropy in addition allows to set the in-batch weights, i.e. make some examples more
	# important than others.
	cross_entropy = tf.losses.sigmoid_cross_entropy(
		logits=logits, multi_class_labels=labels)

	# Create a tensor named cross_entropy for logging purposes.
	tf.identity(cross_entropy, name='cross_entropy')
	tf.summary.scalar('cross_entropy', cross_entropy)

	# If no loss_filter_fn is passed, assume we want the default behavior,
	# which is that batch_normalization variables are excluded from loss.
	def exclude_batch_norm(name):
		return 'batch_normalization' not in name

	loss_filter_fn = loss_filter_fn or exclude_batch_norm

	# Add weight decay to the loss.
	l2_loss = weight_decay * tf.add_n(
		# loss is computed using fp32 for numerical stability.
		[tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
		 if loss_filter_fn(v.name)])
	tf.summary.scalar('l2_loss', l2_loss)
	loss = cross_entropy + l2_loss

	if mode == tf.estimator.ModeKeys.TRAIN:
		global_step = tf.train.get_or_create_global_step()

		learning_rate = learning_rate_fn(global_step)

		# Create a tensor named learning_rate for logging purposes
		tf.identity(learning_rate, name='learning_rate')
		tf.summary.scalar('learning_rate', learning_rate)

		optimizer = tf.train.MomentumOptimizer(
			learning_rate=learning_rate,
			momentum=momentum
		)

		def _dense_grad_filter(gvs):
			"""Only apply gradient updates to the final layer.

	  This function is used for fine tuning.

	  Args:
		gvs: list of tuples with gradients and variable info
	  Returns:
		filtered gradients so that only the dense layer remains
	  """
			return [(g, v) for g, v in gvs if 'dense' in v.name]

		if loss_scale != 1:
			# When computing fp16 gradients, often intermediate tensor values are
			# so small, they underflow to 0. To avoid this, we multiply the loss by
			# loss_scale to make these tensor values loss_scale times bigger.
			scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

			if fine_tune:
				scaled_grad_vars = _dense_grad_filter(scaled_grad_vars)

			# Once the gradient computation is complete we can scale the gradients
			# back to the correct scale before passing them to the optimizer.
			unscaled_grad_vars = [(grad / loss_scale, var)
								  for grad, var in scaled_grad_vars]
			minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
		else:
			grad_vars = optimizer.compute_gradients(loss)
			if fine_tune:
				grad_vars = _dense_grad_filter(grad_vars)
			minimize_op = optimizer.apply_gradients(grad_vars, global_step)

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		train_op = tf.group(minimize_op, update_ops)
	else:
		train_op = None

	p = tf.cast(predictions['k_hot_predictions'], tf.int32)
	b = tf.pow(2, tf.range(0, model.num_classes))
	b = tf.reshape(b,[model.num_classes, 1])
	l = tf.matmul(a=labels,b=b)
	p = tf.matmul(a=p,b=b)
	accuracy = tf.metrics.accuracy(labels=l, predictions=p)

	f1 = tf_f1_score(labels, tf.cast(predictions['k_hot_predictions'], tf.int64))
	counts, update = streaming_counts(labels, tf.cast(predictions['k_hot_predictions'], tf.int64), model.num_classes)
	streamed_f1 = streaming_f1(counts)

	metrics = {
		"accuracy": accuracy,
		"f1_per_class": (streamed_f1[3], update),
	}

	tf.identity(accuracy[1], name='train_accuracy')
	tf.identity(streamed_f1[1], name='f1_cum')
	tf.identity(f1[1], name='f1_overall')

	tf.summary.scalar('train_accuracy', accuracy[1])
	tf.summary.scalar('f1_cum', streamed_f1[1])
	tf.summary.scalar('f1_overall', f1[1])

	return tf.estimator.EstimatorSpec(
		mode=mode,
		predictions=predictions,
		loss=loss,
		train_op=train_op,
		eval_metric_ops=metrics)


def resnet_main(
		flags_obj, model_function, input_function, dataset_name, shape=None):
	"""Shared main loop for ResNet Models.

  Args:
	flags_obj: An object containing parsed flags. See define_resnet_flags()
	  for details.
	model_function: the function that instantiates the Model and builds the
	  ops for data_batch_1.bin/eval. This will be passed directly into the estimator.
	input_function: the function that processes the dataset and returns a
	  dataset that the estimator can dataset on. This will be wrapped with
	  all the relevant flags for running and passed to estimator.
	dataset_name: the name of the dataset for training and evaluation. This is
	  used for logging purpose.
	shape: list of ints representing the shape of the images used for training.
	  This is only used if flags_obj.export_dir is passed.
  """

	model_helpers.apply_clean(flags.FLAGS)

	# Ensures flag override logic is only executed if explicitly triggered.
	if flags_obj.tf_gpu_thread_mode:
		override_flags_and_set_envars_for_gpu_thread_pool(flags_obj)

	# Creates session config. allow_soft_placement = True, is required for
	# multi-GPU and is not harmful for other modes.
	session_config = tf.ConfigProto(
		inter_op_parallelism_threads=flags_obj.inter_op_parallelism_threads,
		intra_op_parallelism_threads=flags_obj.intra_op_parallelism_threads,
		allow_soft_placement=True)

	distribution_strategy = distribution_utils.get_distribution_strategy(
		flags_core.get_num_gpus(flags_obj), flags_obj.all_reduce_alg)

	# Creates a `RunConfig` that checkpoints every 24 hours which essentially
	# results in checkpoints determined only by `epochs_between_evals`.
	run_config = tf.estimator.RunConfig(
		train_distribute=distribution_strategy,
		session_config=session_config,
		save_checkpoints_secs=60 * 60 * 24)

	# Initializes model with all but the dense layer from pretrained ResNet.
	if flags_obj.pretrained_model_checkpoint_path is not None:
		warm_start_settings = tf.estimator.WarmStartSettings(
			flags_obj.pretrained_model_checkpoint_path,
			vars_to_warm_start='^(?!.*dense)')
	else:
		warm_start_settings = None

	classifier = tf.estimator.Estimator(
		model_fn=model_function, model_dir=flags_obj.model_dir, config=run_config,
		warm_start_from=warm_start_settings, params={
			'resnet_size':int(flags_obj.resnet_size),
			'data_format':flags_obj.data_format,
			'batch_size':flags_obj.batch_size,
			'resnet_version':int(flags_obj.resnet_version),
			'loss_scale':flags_core.get_loss_scale(flags_obj),
			'dtype':flags_core.get_tf_dtype(flags_obj),
			'fine_tune':flags_obj.fine_tune
		})

	run_params = {
		'batch_size':flags_obj.batch_size,
		'dtype':flags_core.get_tf_dtype(flags_obj),
		'resnet_size':flags_obj.resnet_size,
		'resnet_version':flags_obj.resnet_version,
		'synthetic_data':flags_obj.use_synthetic_data,
		'train_epochs':flags_obj.train_epochs,
	}
	if flags_obj.use_synthetic_data:
		dataset_name = dataset_name + '-synthetic'

	benchmark_logger = logger.get_benchmark_logger()
	benchmark_logger.log_run_info('resnet', dataset_name, run_params,
								  test_id=flags_obj.benchmark_test_id)

	train_hooks = hooks_helper.get_train_hooks(
		flags_obj.hooks,
		model_dir=flags_obj.model_dir,
		batch_size=flags_obj.batch_size)

	def input_fn_train(num_epochs):
		return input_function(
			is_training=True,
			data_dir=os.path.join(flags_obj.data_dir, 'train'),
			batch_size=distribution_utils.per_device_batch_size(
				flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
			num_epochs=num_epochs,
			dtype=flags_core.get_tf_dtype(flags_obj),
			datasets_num_private_threads=flags_obj.datasets_num_private_threads,
			num_parallel_batches=flags_obj.datasets_num_parallel_batches)

	def input_fn_eval():
		return input_function(
			is_training=False,
			data_dir=os.path.join(flags_obj.data_dir, 'val'),
			batch_size=distribution_utils.per_device_batch_size(
				flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
			num_epochs=1,
			dtype=flags_core.get_tf_dtype(flags_obj))

	if flags_obj.eval_only or not flags_obj.train_epochs:
		# If --eval_only is set, perform a single loop with zero data_batch_1.bin epochs.
		schedule, n_loops = [0], 1
	else:
		# Compute the number of times to loop while training. All but the last
		# pass will data_batch_1.bin for `epochs_between_evals` epochs, while the last will
		# data_batch_1.bin for the number needed to reach `training_epochs`. For instance if
		#   train_epochs = 25 and epochs_between_evals = 10
		# schedule will be set to [10, 10, 5]. That is to say, the loop will:
		#   Train for 10 epochs and then evaluate.
		#   Train for another 10 epochs and then evaluate.
		#   Train for a final 5 epochs (to reach 25 epochs) and then evaluate.
		n_loops = math.ceil(flags_obj.train_epochs / flags_obj.epochs_between_evals)
		schedule = [flags_obj.epochs_between_evals for _ in range(int(n_loops))]
		schedule[-1] = flags_obj.train_epochs - sum(schedule[:-1])  # over counting.

	for cycle_index, num_train_epochs in enumerate(schedule):
		tf.logging.info('Starting cycle: %d/%d', cycle_index, int(n_loops))

		if num_train_epochs:
			classifier.train(input_fn=lambda:input_fn_train(num_train_epochs),
							 hooks=train_hooks, max_steps=flags_obj.max_train_steps)

		tf.logging.info('Starting to evaluate.')

		# flags_obj.max_train_steps is generally associated with testing and
		# profiling. As a result it is frequently called with synthetic data, which
		# will iterate forever. Passing steps=flags_obj.max_train_steps allows the
		# eval (which is generally unimportant in those circumstances) to terminate.
		# Note that eval will run for max_train_steps each loop, regardless of the
		# global_step count.
		eval_results = classifier.evaluate(input_fn=input_fn_eval,
										   steps=flags_obj.max_train_steps)

		pred_results = classifier.predict(input_fn=input_fn_eval)

		pred = []
		for p in pred_results:
			pred.append(np.concatenate((p['logits'], p['k_hot_predictions'], p['labels'])))
		pred = np.array(pred)
		np.savetxt('test_result_{}.csv'.format(flags_obj.model_dir[:-1]), pred, delimiter=',')

		benchmark_logger.log_evaluation_result(eval_results)

		if model_helpers.past_stop_threshold(
				flags_obj.stop_threshold, eval_results['accuracy']):
			break

	if flags_obj.export_dir is not None:
		# Exports a saved model for the given classifier.
		export_dtype = flags_core.get_tf_dtype(flags_obj)
		if flags_obj.image_bytes_as_serving_input:
			input_receiver_fn = functools.partial(
				image_bytes_serving_input_fn, shape, dtype=export_dtype)
		else:
			input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
				shape, batch_size=flags_obj.batch_size, dtype=export_dtype)
		classifier.export_savedmodel(flags_obj.export_dir, input_receiver_fn,
									 strip_default_attrs=True)


def define_resnet_flags(resnet_size_choices=None):
	"""Add flags and validators for ResNet."""
	flags_core.define_base()
	flags_core.define_performance(num_parallel_calls=False,
								  tf_gpu_thread_mode=True,
								  datasets_num_private_threads=True,
								  datasets_num_parallel_batches=True)
	flags_core.define_image()
	flags_core.define_benchmark()
	flags.adopt_module_key_flags(flags_core)

	flags.DEFINE_enum(
		name='resnet_version', short_name='rv', default='2',
		enum_values=['1', '2'],
		help=flags_core.help_wrap(
			'Version of ResNet. (1 or 2) See README.md for details.'))
	flags.DEFINE_bool(
		name='fine_tune', short_name='ft', default=False,
		help=flags_core.help_wrap(
			'If True do not data_batch_1.bin any parameters except for the final layer.'))
	flags.DEFINE_string(
		name='pretrained_model_checkpoint_path', short_name='pmcp', default=None,
		help=flags_core.help_wrap(
			'If not None initialize all the network except the final layer with '
			'these values'))
	flags.DEFINE_boolean(
		name='eval_only', default=False,
		help=flags_core.help_wrap('Skip training and only perform evaluation on '
								  'the latest checkpoint.'))
	flags.DEFINE_boolean(
		name='image_bytes_as_serving_input', default=False,
		help=flags_core.help_wrap(
			'If True exports savedmodel with serving signature that accepts '
			'JPEG image bytes instead of a fixed size [HxWxC] tensor that '
			'represents the image. The former is easier to use for serving at '
			'the expense of image resize/cropping being done as part of model '
			'inference. Note, this flag only applies to ImageNet and cannot '
			'be used for CIFAR.'))

	choice_kwargs = dict(
		name='resnet_size', short_name='rs', default='50',
		help=flags_core.help_wrap('The size of the ResNet model to use.'))

	if resnet_size_choices is None:
		flags.DEFINE_string(**choice_kwargs)
	else:
		flags.DEFINE_enum(enum_values=resnet_size_choices, **choice_kwargs)
