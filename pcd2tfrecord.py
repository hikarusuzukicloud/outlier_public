import os
import sys

import numpy as np
import tensorflow as tf
def normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, np.array([m])

def unit(n):
    m = np.sqrt(np.sum(n**2, axis=1)).reshape(n.shape[0], 1)
    n = n / m
    return n


def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value.tolist()]))
def _byte_list_feature(values):
    value_serialized = tf.io.serialize_tensor(values).numpy()
    return tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[value_serialized]))
"""
def data2tfexample(di):
    ps, labels = di
    num = ps.shape[0]
    feature = {"points":_byte_list_feature(ps),
               "labels":_byte_list_feature(labels)}

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
"""

def data2tfexample(di):
    _, features, labels, centroid, max_dis = di
    num = features.shape[0]
    feature = {"features":_byte_list_feature(features),
               "labels":_byte_list_feature(labels),
               "centroid":_byte_list_feature(centroid),
               "max_dis":_byte_list_feature(max_dis)}

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def main():
    points_dir = "/local_disk/hikaru/pointnet2/catdata/crop32_sampling512/labelingcolor/valdata_repeat_max50_label4"
    label_dir = "/local_disk/hikaru/pointnet2/catdata/crop32_sampling512/labelingcolor/valdata_label_repeat_max50_label4"
    tfrecord_dir = "/local_disk/hikaru/pointnet2/tfrecords/cat_crop32_512_val_repeat_max50_label4"
    #os.makedirs(tfrecord_dir)

    Files = os.listdir(points_dir)
    Dataset = []
    print(len(Files))
    count = 0
    for p in Files:
        fp = os.path.join(points_dir, p)
        pcd_np = np.loadtxt(fp)

        points, colors, normals = np.split(pcd_np, 3, 1) 
        points_normalize, centroid, max_dis = normalize(points)
        normals_unit = unit(normals)
        pcd_np = np.block([points_normalize, colors, normals_unit])

        pcd_np = pcd_np.astype(np.float32)
        points_normalize = points_normalize.astype(np.float32)
        centroid = centroid.astype(np.float32)
        max_dis = max_dis.astype(np.float32)

        fp_label = os.path.join(label_dir, p)
        label_np = np.loadtxt(fp_label)
        label_np = label_np.astype(np.int32)

        Dataset.append([points_normalize, pcd_np, label_np, centroid, max_dis])
        count+=1
        if count % 1000 == 0:
            print(count)

    num_train = len(Dataset)
    NUM_SHARDS = 1
    num_per_shard = int(num_train / NUM_SHARDS)

    random_idx = np.arange(num_train)
    np.random.shuffle(random_idx)
    for shard_id in range(NUM_SHARDS):
        filename = os.path.join(tfrecord_dir, 
        #    "cat_crop32_512_train_%03d-of-%03d.tfrecords" % (shard_id+1, NUM_SHARDS))
            "cat_crop32_512_val_repeat_%03d-of-%03d.tfrecords" % (shard_id+1, NUM_SHARDS))
        start_idx = shard_id * num_per_shard
        end_idx = min((shard_id+1) * num_per_shard, num_train)
        print(filename)
        with tf.io.TFRecordWriter(filename) as tfrecord_writer:
            for i in range(start_idx, end_idx):
                example = data2tfexample(Dataset[random_idx[i]])
                tfrecord_writer.write(example)

if __name__ == '__main__':
    main()
