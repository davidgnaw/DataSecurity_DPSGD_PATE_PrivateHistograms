def batch_indices(batch_nb, data_length, batch_size):
  """
  This helper function computes a batch start and end index
  :param batch_nb: the batch number
  :param data_length: the total length of the data being parsed by batches
  :param batch_size: the number of inputs in each batch
  :return: pair of (start, end) indices
  """
  # Batch start and end index
  start = int(batch_nb * batch_size)
  end = int((batch_nb + 1) * batch_size)

  # When there are not enough inputs left, we reuse some to complete the batch
  if end > data_length:
    shift = end - data_length
    start -= shift
    end -= shift

  return start, end