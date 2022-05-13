from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

import matplotlib.pyplot as plt
import numpy as np


flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 0.1, #sigma 
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 10, 'Batch size') #for sake of run time
flags.DEFINE_integer('epochs', 10, 'Number of epochs') #for sake of run time
flags.DEFINE_integer(
    'microbatches', 10, 'Number of microbatches ' #for sake of run time
    '(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')

FLAGS = flags.FLAGS


def compute_epsilon(steps, sigma):
  """Computes epsilon value for given hyperparameters."""
  if sigma == 0.0:
    return float('inf')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  sampling_probability = FLAGS.batch_size / 60000
  rdp = compute_rdp(
      q=sampling_probability,
      noise_multiplier=sigma, #change
      steps=steps,
      orders=orders)
  # Delta is set to 1e-5 because MNIST has 60000 training points.
  return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]


def load_mnist():
  """Loads MNIST and preprocesses to combine training and validation data."""
  train, test = tf.keras.datasets.mnist.load_data()
  train_data, train_labels = train
  test_data, test_labels = test

  train_data = np.array(train_data, dtype=np.float32) / 255
  test_data = np.array(test_data, dtype=np.float32) / 255

  train_data = train_data.reshape((train_data.shape[0], 28, 28, 1))
  test_data = test_data.reshape((test_data.shape[0], 28, 28, 1))

  train_labels = np.array(train_labels, dtype=np.int32)
  test_labels = np.array(test_labels, dtype=np.int32)

  train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
  test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

  assert train_data.min() == 0.
  assert train_data.max() == 1.
  assert test_data.min() == 0.
  assert test_data.max() == 1.

  return train_data, train_labels, test_data, test_labels


def run(l2, sigma):
  logging.set_verbosity(logging.INFO)
  if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')

  # Load training and test data.
  train_data, train_labels, test_data, test_labels = load_mnist()

  # Define a sequential Keras model
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(
          16,
          8,
          strides=2,
          padding='same',
          activation='relu',
          input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPool2D(2, 1),
      tf.keras.layers.Conv2D(
          32, 4, strides=2, padding='valid', activation='relu'),
      tf.keras.layers.MaxPool2D(2, 1),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(10)
  ])

  if FLAGS.dpsgd:
    optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=l2, #change
        noise_multiplier=sigma, #change
        num_microbatches=FLAGS.microbatches,
        learning_rate=FLAGS.learning_rate)
    # Compute vector of per-example loss rather than its mean over a minibatch.
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.losses.Reduction.NONE)
  else:
    optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

  # Compile model with Keras
  model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

  # Train model with Keras
  hist = model.fit(
      train_data,
      train_labels,
      epochs=FLAGS.epochs,
      validation_data=(test_data, test_labels),
      batch_size=FLAGS.batch_size)

  accuracy = hist.history['accuracy'][-1]
  # Compute the privacy budget expended.
  if FLAGS.dpsgd:
    eps = compute_epsilon(FLAGS.epochs * 60000 // FLAGS.batch_size, sigma)
    #print('For delta=1e-5, the current epsilon is: %.2f' % eps)
    return [eps, accuracy] #return epsilon and accuracy as a list

def main(unused_argv):

  #part 1
  # changing l2 clipping norm and plotting with sigma of 0.1
  accuracy1 = [] # create array of epsilons based on differing l2 clipping norms
  l2_norms = [1.0, 2.0, 3.0, 4.0, 5.0] #iterate 5 times with l2 norms and epsilon

  for i in range(0 ,len(l2_norms)):
    results1 = run(l2_norms[i], 0.1) #keep sigma the same for part 1 but vary the l2 clipping norm
    accuracy1.append(results1[1]) #accuracy is in index 1 

  plt.figure(1)
  plt.plot(l2_norms, accuracy1)
  plt.xlabel("l2 clipping norm")
  plt.ylabel("accuracy")

  plt.savefig('l2 clip vs Accuracy.png')


  #part 2
  #changing sigmas to see how it affects model accuracy and using l2 clip norm = 4.0
  accuracy2 = []
  sigmas = [0.02, 0.05, 0.1, 0.2, 0.3] #changing noise multipler (sigma)

  for i in range(0, len(sigmas)):
    results2 = run(4.0, sigmas[i])
    accuracy2.append(results2[1])

  plt.figure(2)
  plt.plot(sigmas, accuracy2)
  plt.xlabel("sigma")
  plt.ylabel("accuracy")

  plt.savefig('Sigma vs Accuracy.png')

  #part 3
  #generate epsilons based on different sigmas while keeping l2 clipping norm to 4.0
  #noticed that epsilons become < 20 at roughly >0.28 sigma, as such, we will just utilize sigma >0.28 (MIN CASE where epsilon~20 is 0.28)
  #noticed ... (MAX CASE is 0.9)

  #tested by increments of 4
  #realized sigmas are... 0.282 for 20, 0.343 for 10, 0.42 for 5, 0.58 for 2, 0.74 for 1, 0.98 for 0.5, 1.90 for 0.1

  epsilons = []
  sigmas = [0.282, 0.343, 0.42, 0.58, 0.74, 0.98, 1.90]

  

  for i in range(0, len(sigmas)):
    result = run(4.0, sigmas[i])
    epsilons.append(result[0])

  #write results to text file

  with open("part3.txt", "w") as f:

    print("Sigma:Epsilon", file=f)
    for i in range(0, len(epsilons)):
      print(str(sigmas[i]) + ":" + str(epsilons[i]), file=f)



if __name__ == '__main__':
  app.run(main)