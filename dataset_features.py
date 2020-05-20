import tensorflow as tf
from data import dataset_myfun

class TFDataset():

    def __init__(self, path, batch_size, dataset='modelnet', shuffle_buffer=10000):

        self.path = path
        self.batch_size = batch_size
        self.iterator = None
        self.dataset_type = dataset
        self.shuffle_buffer = shuffle_buffer
        self.dataset_num = 0

        self.dataset = self.read_tfrecord(self.path, self.batch_size)

        self.get_iterator()


    def read_tfrecord(self, path, batch_size):
        if self.dataset_type == 'cat':
            dataset = tf.data.TFRecordDataset(path)
            dataset = dataset.map(dataset_myfun.parse_function_with_features)
            dataset = dataset.shuffle(self.shuffle_buffer).batch(batch_size)
            return dataset
        elif self.dataset_type == 'cat_test':
            dataset = tf.data.TFRecordDataset(path)
            self.dataset_num = len(list(dataset))
            dataset = dataset.map(dataset_myfun.parse_function_test)
            dataset = dataset.shuffle(self.shuffle_buffer).batch(batch_size)
            return dataset
        dataset = tf.data.TFRecordDataset(path).shuffle(self.shuffle_buffer).batch(batch_size)

        return dataset


    # VarLenFeature : 可変長
    # FixedLenFeature : 固定長、クラス分類で１つに確定しているから
    def get_iterator(self):

        self.iterator = self.dataset.__iter__()


    def reset_iterator(self):

        self.dataset.shuffle(self.shuffle_buffer)
        self.get_iterator()

    def get_batch(self):

        while True:
            try:
                batch = self.iterator.next()
                features = batch[0]
                label = batch[1]
                break
            except:
                self.reset_iterator()

        return features, label

    def get_batch_test(self):

        while True:
            try:
                batch = self.iterator.next()
                features = batch[0]
                label = batch[1]
                centroid = batch[2]
                max_dis = batch[3]
                break
            except:
                self.reset_iterator()

        return features, label, centroid, max_dis

    def get_dataset_num(self):
        return self.dataset_num
