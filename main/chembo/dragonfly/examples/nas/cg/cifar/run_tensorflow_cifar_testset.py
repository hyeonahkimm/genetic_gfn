# ==============================================================================
# Written by Willie.
# ==============================================================================

"""
 Defining, training, and evaluating neural networks in tensorflow on CIFAR-10.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import os
import argparse
import tensorflow as tf
import cifar.cifar10_myMain as cifar10_myMain
import cifar.cifar10 as cifar10
import numpy as np
from opt.nn_opt_utils import get_initial_pool


os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # To remove the tensorflow compilation warnings
os.environ['TF_SYNC_ON_FINISH'] = '0'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'


def compute_validation_error(nn,data_dir,gpu_id,params):
  """ Trains tensorflow neural network and then computes validation error. """

  # Fixed variables for CIFAR10 CNN runs
  #variable_strategy = 'CPU'; num_gpus = 0 # For CPU
  variable_strategy = 'GPU'; num_gpus = 1 # For GPU
  use_distortion_for_training = True
  log_device_placement = False
  num_intra_threads = 0
  weight_decay = 2e-4
  momentum = 0.9
  learning_rate = params['learningRate']
  data_format = None
  batch_norm_decay = 0.997
  batch_norm_epsilon = 1e-5

  hparams = argparse.Namespace(weight_decay=weight_decay, momentum=momentum,
      learning_rate=learning_rate, data_format=data_format,
      batch_norm_decay=batch_norm_decay, batch_norm_epsilon=batch_norm_epsilon,
      train_batch_size=params['trainBatchSize'],
      eval_batch_size=params['valiBatchSize'], sync=False)

  # Get model_fn
  model_fn = cifar10_myMain.get_model_fn(num_gpus, variable_strategy, 1, nn)
  
  # Set model = tf.estimator.Estimator(model_fn)
  model = tf.estimator.Estimator(model_fn,params=hparams)

  # Define train_input_fn, vali_input_fn, and eval_input_fn
  train_input_fn = functools.partial(
      cifar10_myMain.input_fn,
      data_dir,
      subset='train',
      num_shards=num_gpus,
      batch_size=params['trainBatchSize'],
      use_distortion_for_training=use_distortion_for_training)

  vali_input_fn = functools.partial(
      cifar10_myMain.input_fn,
      data_dir,
      subset='validation',
      batch_size=params['valiBatchSize'],
      num_shards=num_gpus)

  eval_input_fn = functools.partial(
      cifar10_myMain.input_fn,
      data_dir,
      subset='eval',
      batch_size=params['evalBatchSize'], #### NEED TO DEFINE/INCLUDE evalBatchSize
      num_shards=num_gpus)

  # Loop through numLoops: call model.train, and model.evaluate
  neg_vali_errors = []
  neg_eval_errors = []
  for loop in range(params['numLoops']):
    model.train(train_input_fn,steps=params['trainNumStepsPerLoop'])
    neg_vali_errors.append(model.evaluate(vali_input_fn,steps=params['valiNumStepsPerLoop'])['accuracy'])
    neg_eval_errors.append(model.evaluate(eval_input_fn,steps=params['evalNumStepsPerLoop'])['accuracy'])
    print('Finished iter: ' + str((loop+1)*params['trainNumStepsPerLoop']))

  # Print all validation errors and test errors
  print('List of validation accuracies:')
  print(neg_vali_errors)
  print('List of test accuracies:')
  print(neg_eval_errors)

  # Return eval error at index where neg_vali_errors is maximized
  return neg_eval_errors[np.argmax(neg_vali_errors)]
