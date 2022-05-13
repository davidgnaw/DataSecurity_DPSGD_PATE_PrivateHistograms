from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import deep_cnn
import input  # pylint: disable=redefined-builtin
import metrics
import tensorflow.compat.v1 as tf

import matplotlib.pyplot as plt
import numpy as np


tf.flags.DEFINE_string('dataset', 'svhn', 'The name of the dataset to use')
tf.flags.DEFINE_integer('nb_labels', 10, 'Number of output classes')

tf.flags.DEFINE_string('data_dir','/tmp','Temporary storage')
tf.flags.DEFINE_string('train_dir','/tmp/train_dir',
                       'Where model ckpt are saved')

tf.flags.DEFINE_integer('max_steps', 100, 'Number of training steps to run.') #runtime purposes, decrease from 3000 to 100 so we can focus on training all teachers
tf.flags.DEFINE_integer('nb_teachers', 100, 'Teachers in the ensemble.') #100 teachers
tf.flags.DEFINE_integer('teacher_id', 0, 'ID of teacher being trained.')

tf.flags.DEFINE_boolean('deeper', False, 'Activate deeper CNN model')

FLAGS = tf.flags.FLAGS


def train_teacher(dataset, nb_teachers, teacher_id):
  """
  This function trains a teacher (teacher id) among an ensemble of nb_teachers
  models for the dataset specified.
  :param dataset: string corresponding to dataset (svhn, cifar10)
  :param nb_teachers: total number of teachers in the ensemble
  :param teacher_id: id of the teacher being trained
  :return: True if everything went well
  """
  # If working directories do not exist, create them
  assert input.create_dir_if_needed(FLAGS.data_dir)
  assert input.create_dir_if_needed(FLAGS.train_dir)

  # Load the dataset
  if dataset == 'svhn':
    train_data,train_labels,test_data,test_labels = input.ld_svhn(extended=True)
  elif dataset == 'cifar10':
    train_data, train_labels, test_data, test_labels = input.ld_cifar10()
  elif dataset == 'mnist':
    train_data, train_labels, test_data, test_labels = input.ld_mnist()
  else:
    print("Check value of dataset flag")
    return False

  # Retrieve subset of data for this teacher
  data, labels = input.partition_dataset(train_data,
                                         train_labels,
                                         nb_teachers,
                                         teacher_id)

  print("Length of training data: " + str(len(labels)))

  # Define teacher checkpoint filename and full path
  if FLAGS.deeper:
    filename = str(nb_teachers) + '_teachers_' + str(teacher_id) + '_deep.ckpt'
  else:
    filename = str(nb_teachers) + '_teachers_' + str(teacher_id) + '.ckpt'
  ckpt_path = FLAGS.train_dir + '/' + str(dataset) + '_' + filename

  # Perform teacher training
  assert deep_cnn.train(data, labels, ckpt_path)

  # Append final step value to checkpoint for evaluation
  ckpt_path_final = ckpt_path + '-' + str(FLAGS.max_steps - 1)

  # Retrieve teacher probability estimates on the test data
  teacher_preds = deep_cnn.softmax_preds(test_data, ckpt_path_final)

  # Compute teacher accuracy
  precision = metrics.accuracy(teacher_preds, test_labels)
  #print('Precision of teacher after training: ' + str(precision))
  #return True

  #return precision instead of printing it and asserting boolean
  return precision


def main(argv=None):  # pylint: disable=unused-argument
  # Make a call to train_teachers with values specified in flags

  # part 1
  accuracy_10 = [] # for graphing
  accuracy_50 = []
  accuracy_100 = []

  num_teachers_10 = []
  num_teachers_50 = []
  num_teachers_100 = []

  #teacher ID ranges from 0 to 99 so 100 teachers in total (too much so will just limit it for now)

  #case of 10 teachers

  for i in range(0, 10):
    num_teachers_10.append(10)
    accuracy_10.append(train_teacher(FLAGS.dataset, 10, i)) # i is teacher ID

  #case of 50 teachers

  for i in range(0, 50):
    num_teachers_50.append(50)
    accuracy_50.append(train_teacher(FLAGS.dataset, 50, i))

  #case of 100 teachers

  for i in range(0, 100):
    num_teachers_100.append(100)
    accuracy_100.append(train_teacher(FLAGS.dataset, 100, i))

  # plot out num teachers vs accuracy
  plt.figure(1)
  plt.scatter(num_teachers_10, accuracy_10)
  plt.scatter(num_teachers_50, accuracy_50)
  plt.scatter(num_teachers_100, accuracy_100)
  plt.xticks([10, 50, 100])
  plt.xlabel("number of teachers")
  plt.ylabel("accuracy")

  plt.savefig("Number of Teachers vs Accuracy.png")






if __name__ == '__main__':
  tf.app.run()