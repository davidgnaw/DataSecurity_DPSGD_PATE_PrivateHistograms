from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import aggregation
import deep_cnn
import input  # pylint: disable=redefined-builtin
import metrics
import numpy as np
from six.moves import xrange
import tensorflow.compat.v1 as tf

import statistics #to get aggregated accuracy
import matplotlib.pyplot as plt
import numpy as np


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('dataset', 'svhn', 'The name of the dataset to use')
tf.flags.DEFINE_integer('nb_labels', 10, 'Number of output classes')

tf.flags.DEFINE_string('data_dir','/tmp','Temporary storage')
tf.flags.DEFINE_string('train_dir','/tmp/train_dir','Where model chkpt are saved')
tf.flags.DEFINE_string('teachers_dir','/tmp/train_dir',
                       'Directory where teachers checkpoints are stored.')

tf.flags.DEFINE_integer('teachers_max_steps', 100,
                        'Number of steps teachers were ran.')
tf.flags.DEFINE_integer('max_steps', 100, 'Number of steps to run student.')
tf.flags.DEFINE_integer('nb_teachers', 100, 'Teachers in the ensemble.')
tf.flags.DEFINE_integer('stdnt_share', 1000,
                        'Student share (last index) of the test data')
tf.flags.DEFINE_integer('lap_scale', 10,
                        'Scale of the Laplacian noise added for privacy')
tf.flags.DEFINE_boolean('save_labels', False,
                        'Dump numpy arrays of labels and clean teacher votes')
tf.flags.DEFINE_boolean('deeper', False, 'Activate deeper CNN model')


def ensemble_preds(dataset, nb_teachers, stdnt_data):
  """
  Given a dataset, a number of teachers, and some input data, this helper
  function queries each teacher for predictions on the data and returns
  all predictions in a single array. (That can then be aggregated into
  one single prediction per input using aggregation.py (cf. function
  prepare_student_data() below)
  :param dataset: string corresponding to mnist, cifar10, or svhn
  :param nb_teachers: number of teachers (in the ensemble) to learn from
  :param stdnt_data: unlabeled student training data
  :return: 3d array (teacher id, sample id, probability per class)
  """

  # Compute shape of array that will hold probabilities produced by each
  # teacher, for each training point, and each output class
  result_shape = (nb_teachers, len(stdnt_data), FLAGS.nb_labels)

  # Create array that will hold result
  result = np.zeros(result_shape, dtype=np.float32)

  # Get predictions from each teacher
  for teacher_id in xrange(nb_teachers):
    # Compute path of checkpoint file for teacher model with ID teacher_id
    if FLAGS.deeper:
      ckpt_path = FLAGS.teachers_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_teachers_' + str(teacher_id) + '_deep.ckpt-' + str(FLAGS.teachers_max_steps - 1) #NOLINT(long-line)
    else:
      ckpt_path = FLAGS.teachers_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_teachers_' + str(teacher_id) + '.ckpt-' + str(FLAGS.teachers_max_steps - 1)  # NOLINT(long-line)

    # Get predictions on our training data and store in result array
    result[teacher_id] = deep_cnn.softmax_preds(stdnt_data, ckpt_path)

    # This can take a while when there are a lot of teachers so output status
    print("Computed Teacher " + str(teacher_id) + " softmax predictions")

  return result


def prepare_student_data(dataset, nb_teachers, lap_scale, save=False):
  """
  Takes a dataset name and the size of the teacher ensemble and prepares
  training data for the student model, according to parameters indicated
  in flags above.
  :param dataset: string corresponding to mnist, cifar10, or svhn
  :param nb_teachers: number of teachers (in the ensemble) to learn from
  :param save: if set to True, will dump student training labels predicted by
               the ensemble of teachers (with Laplacian noise) as npy files.
               It also dumps the clean votes for each class (without noise) and
               the labels assigned by teachers
  :return: pairs of (data, labels) to be used for student training and testing
  """
  assert input.create_dir_if_needed(FLAGS.train_dir)

  # Load the dataset
  if dataset == 'svhn':
    test_data, test_labels = input.ld_svhn(test_only=True)
  elif dataset == 'cifar10':
    test_data, test_labels = input.ld_cifar10(test_only=True)
  elif dataset == 'mnist':
    test_data, test_labels = input.ld_mnist(test_only=True)
  else:
    print("Check value of dataset flag")
    return False

  # Make sure there is data leftover to be used as a test set
  assert FLAGS.stdnt_share < len(test_data)

  # Prepare [unlabeled] student training data (subset of test set)
  stdnt_data = test_data[:FLAGS.stdnt_share]

  # Compute teacher predictions for student training data
  teachers_preds = ensemble_preds(dataset, nb_teachers, stdnt_data)

  # Aggregate teacher predictions to get student training labels
  if not save:
    stdnt_labels = aggregation.noisy_max(teachers_preds, lap_scale)
  else:
    # Request clean votes and clean labels as well
    stdnt_labels, clean_votes, labels_for_dump = aggregation.noisy_max(teachers_preds, lap_scale, return_clean_votes=True) #NOLINT(long-line)

    # Prepare filepath for numpy dump of clean votes
    filepath = FLAGS.data_dir + "/" + str(dataset) + '_' + str(nb_teachers) + '_student_clean_votes_lap_' + str(lap_scale) + '.npy'  # NOLINT(long-line)

    # Prepare filepath for numpy dump of clean labels
    filepath_labels = FLAGS.data_dir + "/" + str(dataset) + '_' + str(nb_teachers) + '_teachers_labels_lap_' + str(lap_scale) + '.npy'  # NOLINT(long-line)

    # Dump clean_votes array
    with tf.gfile.Open(filepath, mode='w') as file_obj:
      np.save(file_obj, clean_votes)

    # Dump labels_for_dump array
    with tf.gfile.Open(filepath_labels, mode='w') as file_obj:
      np.save(file_obj, labels_for_dump)

  # Print accuracy of aggregated labels
  ac_ag_labels = metrics.accuracy(stdnt_labels, test_labels[:FLAGS.stdnt_share])
  print("Accuracy of the aggregated labels: " + str(ac_ag_labels))

  # Store unused part of test set for use as a test set after student training
  stdnt_test_data = test_data[FLAGS.stdnt_share:]
  stdnt_test_labels = test_labels[FLAGS.stdnt_share:]

  if save:
    # Prepare filepath for numpy dump of labels produced by noisy aggregation
    filepath = FLAGS.data_dir + "/" + str(dataset) + '_' + str(nb_teachers) + '_student_labels_lap_' + str(lap_scale) + '.npy' #NOLINT(long-line)

    # Dump student noisy labels array
    with tf.gfile.Open(filepath, mode='w') as file_obj:
      np.save(file_obj, stdnt_labels)

  return stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels


def train_student(dataset, nb_teachers, lap_scale):
  """
  This function trains a student using predictions made by an ensemble of
  teachers. The student and teacher models are trained using the same
  neural network architecture.
  :param dataset: string corresponding to mnist, cifar10, or svhn
  :param nb_teachers: number of teachers (in the ensemble) to learn from
  :return: True if student training went well
  """
  assert input.create_dir_if_needed(FLAGS.train_dir)

  # Call helper function to prepare student data using teacher predictions
  stdnt_dataset = prepare_student_data(dataset, nb_teachers, lap_scale, save=True)

  # Unpack the student dataset
  stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels = stdnt_dataset

  # Prepare checkpoint filename and path
  if FLAGS.deeper:
    ckpt_path = FLAGS.train_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_student_deeper.ckpt' #NOLINT(long-line)
  else:
    ckpt_path = FLAGS.train_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_student.ckpt'  # NOLINT(long-line)

  # Start student training
  assert deep_cnn.train(stdnt_data, stdnt_labels, ckpt_path)

  # Compute final checkpoint name for student (with max number of steps)
  ckpt_path_final = ckpt_path + '-' + str(FLAGS.max_steps - 1)

  # Compute student label predictions on remaining chunk of test set
  student_preds = deep_cnn.softmax_preds(stdnt_test_data, ckpt_path_final)

  # Compute teacher accuracy
  precision = metrics.accuracy(student_preds, stdnt_test_labels)

  return precision

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

def main(argv=None): # pylint: disable=unused-argument
  # Run student training according to values specified in flags

  #part 2

  total_accuracy_teachers = []

  accuracy_students = []
  accuracy_teachers = []

  noise_scale = [0, 10, 20, 30]

  # two accuracies tested for all teachers from one to one hundred

  for i in range(0, len(noise_scale)):
  # first get aggregated accuracy of teachers
    for j in range(0, 100):
      total_accuracy_teachers.append(train_teacher(FLAGS.dataset, 100, j))
    # aggregate teachers accuracy
    accuracy_teachers.append(statistics.mean(total_accuracy_teachers))

    # next get accuracy of students based on noise scale and 100 teachers
    accuracy_students.append(train_student(FLAGS.dataset, 100, noise_scale[i]))


  # plot out num teachers vs accuracy
  plt.figure(1)
  t = plt.plot(noise_scale, accuracy_teachers, label = "teachers")
  s = plt.plot(noise_scale, accuracy_students, label = "students")

  plt.legend()

  plt.xticks([0, 10, 20, 30])
  plt.xlabel("Laplace Noise")
  plt.ylabel("Accuracy")

  plt.savefig("Laplace Noise vs Teacher and Student Accuracy for 100 Queries.png")



if __name__ == '__main__':
  tf.app.run()




