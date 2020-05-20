import os
import sys
import datetime

sys.path.insert(0, './')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
import tensorflow as tf
import open3d as o3d

from models.cls_msg_model import CLS_MSG_Model
from models.cls_ssg_model import CLS_SSG_Model
from models.cls_basic_model import Pointnet_Model
from models.sem_seg_model import SEM_SEG_Model
#from data.dataset import TFDataset
from data.dataset_features import TFDataset

tf.random.set_seed(42)


def train_step(optimizer, model, loss_object, train_loss, train_acc, 
                train_features, train_labels):

    with tf.GradientTape() as tape:
        pred = model(train_features)
        loss = loss_object(train_labels, pred)

    train_loss.update_state([loss])
    train_acc.update_state(train_labels, pred)

    return pred, train_loss, train_acc

def dic2pcd(dic):
    p = np.array(list(dic.keys()))
    cn = np.array((list(dic.values())))
    c = cn[:, 0]
    n = cn[:, 1]
    label = cn[:, 2]
    true_index = []
    for x in label:
        true_index.append(np.argmax(x))
    true_index = np.array(true_index)
    true_index = np.where(true_index == 1)
    #print(c[0])
    #print(n[0])
    #print(c.shape)
    #print(n.shape)


    pcd_fromnp = o3d.geometry.PointCloud()
    pcd_fromnp.points = o3d.utility.Vector3dVector(p[true_index])
    pcd_fromnp.colors = o3d.utility.Vector3dVector(c[true_index])
    pcd_fromnp.normals = o3d.utility.Vector3dVector(n[true_index])
    return pcd_fromnp

def train(config, params):

    model = SEM_SEG_Model(params['batch_size'], params['num_points'], params['num_classes'], params['bn'])
    model.build(input_shape=(params['batch_size'], params['num_points'], 9))
    print(model.summary())
    print('[info] model training...')

    optimizer = tf.keras.optimizers.Adam(lr=params['lr'])
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    train_loss = tf.keras.metrics.Mean()
    test_loss = tf.keras.metrics.Mean()
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    dataset_path = [os.path.join(config["test_tfrecords_dir"], p) 
                    for p in os.listdir(config["test_tfrecords_dir"])]
    #train_ds = TFDataset(config['test_tfrecords'], params['batch_size'], config['dataset_name'])
    train_ds = TFDataset(dataset_path, params['batch_size'], config['dataset_name'])

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    model.load_weights(config["checkpointpath_load"])

    dataset_num = train_ds.dataset_num

    steps = int(dataset_num / params["batch_size"]) + 1
    print(steps)
    dic = {}
    for epoch in range(steps):
        train_features, train_labels, centroid, max_dis = train_ds.get_batch_test()

        pred, loss, train_acc = train_step(
                optimizer,
                model,
                loss_object,
                train_loss,
                train_acc,
                train_features,
                train_labels
        )
        pred_np = pred.numpy()
        train_features_np = train_features.numpy() # (batch, 512, 9)
        centroid_np = centroid.numpy()
        max_dis_np = max_dis.numpy()
        size0 = train_features_np.shape[0]
        size1 = train_features_np.shape[1]
        if (size0 < 1) or (size1 < 1):
            print("error", epoch)
            continue

        for b in range(size0):
            for i in range(size1):
                features_one = train_features_np[b][i]
                p = tuple(np.round(features_one[:3] * max_dis_np[b] + centroid_np[b], 1))
                if p not in dic.keys():
                    dic[p] = [features_one[3:6], features_one[6:], pred_np[b][i]]
                else:
                    outlier_pred = min(dic[p][2][0], pred_np[b][i][0])
                    inlier_pred = max(dic[p][2][1], pred_np[b][i][1])
                    dic[p][2] = np.array([outlier_pred, inlier_pred])
        if epoch % 1000 == 0:
            print(epoch)

    print("recon")
    pcd_fromnp = dic2pcd(dic)
    o3d.io.write_point_cloud("cat_test_result_batch_repeat_repeat30k_round1.ply", pcd_fromnp)

if __name__ == '__main__':
    config = {
            #'test_tfrecords': '/local_disk/hikaru/pointnet2/tfrecords/cat_crop32_512_test/cat_crop32_512_test_001-of-001.tfrecords',
            'test_tfrecords_dir': '/local_disk/hikaru/pointnet2/tfrecords/cat_crop32_512_test_all/',
            'dataset_name' : "cat_test",
            "checkpointpath_load": '/local_disk/hikaru/pointnet2/checkpoints/cat_batch16_repeat/checkpoint_cat_00030000',
    }

    params = {
            'batch_size' : 16,
            'num_points' : 512,
            'num_classes' : 2,
            'lr' : 0.001,
            'msg' : False,
            'bn' : False
    }

    train(config, params)
