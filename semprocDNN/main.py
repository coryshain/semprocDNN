import sys
import math
from data import Data
from semproc_io import dump_model
import argparse
import numpy as np
import pandas as pd
import os



def dump_embeddings():
    dump_model(embedding_weights, 'embedding_weights.txt', 'E', d.F.ix2condition, range(1,embedding_dim+1), session=sess)

def dump_weights():
    dump_model(weights_final, 'weights.txt', 'F', range(1,embedding_dim+1), d.F.ix2consequent, session=sess)


argparser = argparse.ArgumentParser('''
Trains and evaluates a DNN semproc model from counts file.
''')
argparser.add_argument('path', help='Path to data file')
argparser.add_argument('-m', '--mlr', action='store_true', help='Flag indicating fitting of full MLR model (no embedding layer)')
args = argparser.parse_args()

mlr = args.mlr
if mlr:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

d = Data(args.path)
X, y, w = d.F.to_matrices()

batch_size = 256
embedding_dim = 256
depth = 1

sess = tf.Session()
with sess.as_default():
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)


    X_ = tf.sparse_placeholder(tf.float32, shape=[None, d.F.n_condition])
    y_ = tf.placeholder(tf.int32, shape=[None])
    w_ = tf.placeholder(tf.float32, [None])
   
    if mlr:
        bias_final = tf.Variable(tf.random_normal([], stddev=0.1), name='bias_final', dtype=tf.float32)
        weights_final = tf.Variable(tf.random_normal([d.F.n_condition, d.F.n_consequent], stddev=0.1), name='weights_final', dtype=tf.float32)
        activity_final = bias_final + tf.sparse_tensor_dense_matmul(X_, weights_final)
        
    else:
        embedding_weights = tf.Variable(tf.random_normal([d.F.n_condition, embedding_dim], stddev=0.1), name='embeddings')
        embedding = tf.sparse_tensor_dense_matmul(X_, embedding_weights)

        layers = [embedding]

        bias = [None] * depth
        weights = [None] * depth
        activity = [None] * depth
        activation = [None] * depth

        for i in range(depth-1):
            bias[i] = tf.Variable(tf.random_normal([], stddev=0.1), name='bias_%d'%i, dtype=tf.float32)
            weights[i] = tf.Variable(tf.random_normal([embedding_dim, embedding_dim], stddev=0.1), name='weights_%d'%i, dtype=tf.float32)
            activity[i] = bias[i] + tf.matmul(embedding, weights[i])
            activation[i] = tf.nn.relu(activity[i], name='activation_%d'%i)
            layers.append(activation)

        bias_final = tf.Variable(tf.random_normal([], stddev=0.1), name='bias_final', dtype=tf.float32)
        weights_final = tf.Variable(tf.random_normal([embedding_dim, d.F.n_consequent], stddev=0.1), name='weights_final', dtype=tf.float32)
        activity_final = bias_final + tf.matmul(layers[-1], weights_final)

    logits = activity_final
    loss_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=logits)
    weighted_loss_per_example = loss_per_example * w_
    loss_op = tf.reduce_mean(weighted_loss_per_example)

    soft_predictions = tf.nn.softmax(logits)
    hard_predictions = tf.cast(tf.argmax(soft_predictions, axis=1), tf.int32)
    weighted_accuracy_per_example = tf.cast(tf.equal(y_, hard_predictions), tf.float32) * w_
    acc_op = tf.reduce_sum(weighted_accuracy_per_example) / tf.reduce_sum(w_)

    global_step = tf.Variable(0, trainable=False)

    opt = tf.contrib.opt.NadamOptimizer()
    train_op = opt.minimize(
        loss=loss_op,
        global_step=global_step
    )

sess.run(tf.global_variables_initializer())

iteration = 0
n = X.shape[0]
p = np.arange(n)
n_batch = math.ceil(n/batch_size)


while iteration < 1000:
    sys.stderr.write('\n' + '='*50 + '\n')
    sys.stderr.write('Iteration %d\n\n' %(iteration + 1))

    loss = 0
    acc = 0
    np.random.shuffle(p)

    pb = tf.contrib.keras.utils.Progbar(n_batch)

    for i in range(0, n, batch_size):
        batch_len = min(batch_size, n-i)
        X_coo = X[p[i:i+batch_size]].tocoo()
        X_batch = tf.SparseTensorValue(
            indices=np.stack([X_coo.row, X_coo.col], axis=1),
            values=X_coo.data,
            dense_shape=X_coo.shape
        )

        y_batch = y[p[i:i+batch_size]]
        w_batch = w[p[i:i+batch_size]]

        fd = {
            X_: X_batch,
            y_: y_batch,
            w_: w_batch
        }

        _, loss_batch, acc_batch = sess.run([train_op, loss_op, acc_op], feed_dict=fd)
        pb.update((i/batch_size)+1, values=[('loss', loss_batch), ('acc', acc_batch)])
    
    iteration += 1


