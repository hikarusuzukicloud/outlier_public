import os
import sys
import datetime

sys.path.insert(0, './')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
import tensorflow as tf

from models.cls_msg_model import CLS_MSG_Model
from models.cls_ssg_model import CLS_SSG_Model
from models.cls_basic_model import Pointnet_Model
from models.sem_seg_model import SEM_SEG_Model
#from models.model_my1 import SEM_SEG_Model
from data.dataset_features import TFDataset

tf.random.set_seed(42)


def train_step(optimizer, model, loss_object, train_loss, train_acc, 
                train_features, train_labels):

    #onehot = tf.one_hot(train_labels, 2, on_value=1.0, off_value=0.0, axis=-1)
    with tf.GradientTape() as tape:
        # classのcallが呼び出される
        # 入力による計算が行われる
        pred = model(train_features)
        """
        sub = tf.math.subtract(onehot, pred)
        sub = tf.math.abs(sub)
        pred = tf.math.multiply(pred, d)
        loss = tf.math.reduce_mean(pred)
        #ptrd = tf.clip_by_value(pred, 0, 1)
        """
        loss = loss_object(train_labels, pred)

    train_loss.update_state([loss])
    train_acc.update_state(train_labels, pred)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return train_loss, train_acc


def test_step(optimizer, model, loss_object, test_loss, test_acc, test_features, test_labels):

    with tf.GradientTape() as tape:

            pred = model(test_features)
            loss = loss_object(test_labels, pred)

    test_loss.update_state([loss])
    test_acc.update_state(test_labels, pred)

    return test_loss, test_acc


def train(config, params):

    model = SEM_SEG_Model(params['batch_size'], params['num_points'], params['num_classes'], params['bn'])

    # 入力に依存しない初期化=重みなどの初期化
    #model.build(input_shape=(params['batch_size'], params['num_points'], 3))
    model.build(input_shape=(params['batch_size'], params['num_points'], 9))
    print(model.summary())
    print('[info] model training...')

    optimizer = tf.keras.optimizers.Adam(lr=params['lr'])
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    train_loss = tf.keras.metrics.Mean()
    test_loss = tf.keras.metrics.Mean()
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    dataset_train_path = [os.path.join(config['dataset_dir'], p) for p in os.listdir(config['dataset_dir'])]
    train_ds = TFDataset(dataset_train_path, params['batch_size'], config['dataset_name'])
    val_ds = TFDataset(config['test_tfrecords'], params['batch_size'], config['dataset_name'])
    train_summary_writer = tf.summary.create_file_writer(
            os.path.join(config['log_dir'], config['log_code'], 'train')
    )

    test_summary_writer = tf.summary.create_file_writer(
            os.path.join(config['log_dir'], config['log_code'], 'test')
    )
    """
    a = np.ones((16, 512, 1), dtype=float) * 4
    b = np.ones((16, 512, 1), dtype=float)
    c = np.concatenate([a,b],2)
    d = tf.constant(c, dtype=tf.float32)
    """

    template = 'Epoch {}\n Loss: {}, Accuracy: {}\n Test Loss: {}, Test Accuracy: {}'
    if config["weight_load"]:
        model.load_weights(config["checkpointpath_load"])
    #for epoch in range(1000):
    while True:
        train_features, train_labels = train_ds.get_batch()

        loss, train_acc = train_step(
                optimizer,
                model,
                loss_object,
                train_loss,
                train_acc,
                train_features,
                train_labels
        )
        """
        if optimizer.iterations % 10000 == 0:
            optimizer.lr = optimizer.lr - optimizer.lr / 4
            print(optimizer.lr)
        """

        if optimizer.iterations % config['log_freq'] == 0:
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=optimizer.iterations)
                tf.summary.scalar('accuracy', train_acc.result(), step=optimizer.iterations)

        if optimizer.iterations % config['test_freq'] == 0:

            test_features, test_labels = val_ds.get_batch()

            test_loss, test_acc = test_step(
                    optimizer,
                    model,
                    loss_object,
                    test_loss,
                    test_acc,
                    test_features,
                    test_labels
            )

            with test_summary_writer.as_default():

                tf.summary.scalar('loss', test_loss.result(), step=optimizer.iterations)
                tf.summary.scalar('accuracy', test_acc.result(), step=optimizer.iterations)

            print(template.format(int(optimizer.iterations),
                                train_loss.result(), 
                                train_acc.result()*100,
                                test_loss.result(), 
                                test_acc.result()*100))
            if optimizer.iterations % 2000 == 0:
                checkpointpath = "checkpoint_cat_%08d" % (optimizer.iterations)
                checkpointpath = os.path.join(config["checkpointdir"], checkpointpath)
                model.save_weights(checkpointpath)
                if optimizer.iterations % 400000 == 0:
                    break
        #train_loss.reset_states()
        #test_loss.reset_states()
        #train_acc.reset_states()
        #test_acc.reset_states()

if __name__ == '__main__':
    name = "cat_batch16_repeat_label4"
    #name = "test2"

    config = {
            'dataset_dir' : '/local_disk/hikaru/pointnet2/tfrecords/cat_crop32_512_train_repeat_max50_label4',
            'test_tfrecords': '/local_disk/hikaru/pointnet2/tfrecords/cat_crop32_512_val_repeat_max50_label4/cat_crop32_512_val_repeat_001-of-001.tfrecords',
            'dataset_name' : "cat",
            'log_dir' : '/local_disk/hikaru/pointnet2/logs',
            'log_code' : name,
            'log_freq' : 500,
            'test_freq' : 250,
            "checkpointdir": '/local_disk/hikaru/pointnet2/checkpoints/'+name,
            "weight_load": False,
            "checkpointpath_load": '/local_disk/hikaru/pointnet2/checkpoints/cat_batch16_repeat/checkpoint_cat_00030000',
    }
    os.makedirs('/local_disk/hikaru/pointnet2/checkpoints/'+name, exist_ok=True)

    params = {
            'batch_size' : 16,
            'num_points' : 512,
            'num_classes' : 4,
            'lr' : 1e-3,
            'msg' : False,
            'bn' : False
    }

    train(config, params)
