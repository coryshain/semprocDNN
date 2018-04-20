import sys
import math
from data import Data
from semproc_io import dump_model
import argparse
import numpy as np
import os



def dump_embeddings(path, embdim):
    dump_model(embedding_weights, path + '/embedding_weights.txt', 'E', d.F.ix2condition, range(embdim), session=sess)

def dump_weights(path, embdim):
    dump_model(weights_final, path + '/weights.txt', 'F', range(embdim), d.F.ix2consequent, session=sess)

def report_n_params(sess):
    with sess.as_default():
        with sess.graph.as_default():
            n_params = 0
            var_names = [v.name for v in tf.trainable_variables()]
            var_vals = sess.run(tf.trainable_variables())
            sys.stderr.write('Trainable variables:\n')
            for i in range(len(var_names)):
                v_name = var_names[i]
                v_val = var_vals[i]
                cur_params = np.prod(np.array(v_val).shape)
                n_params += cur_params
                sys.stderr.write('  ' + v_name.split(':')[0] + ': %s\n' % str(cur_params))
            sys.stderr.write('Network contains %d total trainable parameters.\n' % n_params)
            sys.stderr.write('\n')


argparser = argparse.ArgumentParser('''
Trains and evaluates a DNN semproc model from counts file.
''')
argparser.add_argument('path', help='Path to data file')
argparser.add_argument('-m', '--mlr', action='store_true', help='Flag indicating fitting of full MLR model (no embedding layer)')
argparser.add_argument('-c', '--cpuonly', action='store_true', help='Do not use GPU if available')
argparser.add_argument('-d', '--dev', type=str, default=None, help='Path to dev (cross-validation) data, if used.')
argparser.add_argument('-o', '--outdir', type=str, default='./semprocDNN_output/', help='Path to output directory.')
argparser.add_argument('-e', '--embdim', type=int, default=256, help='Embedding dimensions')
argparser.add_argument('-l', '--l1reg', type=float, default=0., help='L1 regularization constant')
argparser.add_argument('-L', '--l2reg', type=float, default=0., help='L2 regularization constant')
argparser.add_argument('-E', '--nexamples', type=int, default=0, help='Number of prediction examples to show at each iteration (defaults to 0).')
args = argparser.parse_args()

mlr = args.mlr
cpu = args.cpuonly
if cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

d = Data(args.path)
X, y, w, index = d.F.to_matrices()
if args.dev is not None:
    X_cv, y_cv, w_cv, index_cv = d.encode_data(args.dev, 'F')

batch_size = 256
depth = 1

sess = tf.Session()
with sess.as_default():
    with sess.graph.as_default():
        X_ = tf.sparse_placeholder(tf.float32, shape=[None, d.F.n_condition])
        y_ = tf.placeholder(tf.int32, shape=[None])
        w_ = tf.placeholder(tf.float32, [None])

        l1 = tf.contrib.layers.l1_regularizer(scale=args.l1reg)
        l2 = tf.contrib.layers.l2_regularizer(scale=args.l2reg)
        regularizer_losses = []

        if mlr:
            weights_final = tf.Variable(tf.random_normal([d.F.n_condition, d.F.n_consequent], stddev=0.1), name='weights_final', dtype=tf.float32)
            if args.l1reg > 0:
                regularizer_losses.append(tf.contrib.layers.apply_regularization(l1, [weights_final]))
            if args.l2reg > 0:
                regularizer_losses.append(tf.contrib.layers.apply_regularization(l2, [weights_final]))
            activity_final = tf.sparse_tensor_dense_matmul(X_, weights_final)

        else:
            embedding_weights = tf.Variable(tf.random_normal([d.F.n_condition, args.embdim], stddev=0.1), name='embeddings')
            if args.l1reg > 0:
                regularizer_losses.append(tf.contrib.layers.apply_regularization(l1, [embedding_weights]))
            if args.l2reg > 0:
                regularizer_losses.append(tf.contrib.layers.apply_regularization(l2, [embedding_weights]))
            embedding = tf.sparse_tensor_dense_matmul(X_, embedding_weights)

            layers = [embedding]

            weights = [None] * depth
            activity = [None] * depth
            activation = [None] * depth

            for i in range(depth-1):
                weights[i] = tf.Variable(tf.random_normal([args.embdim, args.embdim], stddev=0.1), name='weights_%d'%i, dtype=tf.float32)
                if args.l1reg > 0:
                    regularizer_losses.append(tf.contrib.layers.apply_regularization(l1, [weights[i]]))
                if args.l2reg > 0:
                    regularizer_losses.append(tf.contrib.layers.apply_regularization(l2, [weights[i]]))
                activity[i] = tf.matmul(embedding, weights[i])
                activation[i] = tf.nn.relu(activity[i], name='activation_%d'%i)
                layers.append(activation)

            weights_final = tf.Variable(tf.random_normal([args.embdim, d.F.n_consequent], stddev=0.1), name='weights_final', dtype=tf.float32)
            if args.l1reg > 0:
                regularizer_losses.append(tf.contrib.layers.apply_regularization(l1, [weights_final]))
            if args.l2reg > 0:
                regularizer_losses.append(tf.contrib.layers.apply_regularization(l2, [weights_final]))
            activity_final = tf.matmul(layers[-1], weights_final)

        logits = activity_final
        loss_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=logits)
        weighted_loss_per_example = loss_per_example * w_
        loss_op = tf.reduce_mean(weighted_loss_per_example)
        if args.l1reg > 0 or args.l2reg > 0:
            loss_op += tf.add_n(regularizer_losses)

        soft_predictions = tf.nn.softmax(logits)
        hard_predictions = tf.cast(tf.argmax(soft_predictions, axis=1), tf.int32)
        weighted_accuracy_per_example = tf.cast(tf.equal(y_, hard_predictions), tf.float32) * w_
        acc_op = tf.reduce_sum(weighted_accuracy_per_example) / tf.reduce_sum(w_)

        global_step = tf.Variable(0, trainable=False)
        incr_global_step = tf.assign(global_step, global_step + 1)
        global_batch_step = tf.Variable(0, trainable=False)

        opt = tf.contrib.opt.NadamOptimizer()
        train_op = opt.minimize(
            loss=loss_op,
            global_step=global_batch_step
        )
        saver = tf.train.Saver()

        loss_total_summary = tf.placeholder(shape=[], dtype='float32', name='loss_total')
        acc_total_summary = tf.placeholder(shape=[], dtype='float32', name='acc_total')

        tf.summary.scalar('loss', loss_total_summary)
        tf.summary.scalar('acc', acc_total_summary)

        summary = tf.summary.merge_all()

        writer_train = tf.summary.FileWriter(args.outdir + '/tensorboard/train')
        writer_cv = tf.summary.FileWriter(args.outdir + '/tensorboard/cv')

        sess.run(tf.global_variables_initializer())

        if os.path.exists(args.outdir + '/checkpoint'):
            saver.restore(sess, args.outdir + '/model.ckpt')



        n = X.shape[0]
        p = np.arange(n)
        n_batch = math.ceil(n/batch_size)

        report_n_params(sess)

        while global_step.eval(session=sess) < 1000:
            iteration = global_step.eval(session=sess)
            sys.stderr.write('\n' + '='*50 + '\n')
            sys.stderr.write('Iteration %d\n\n' %(iteration + 1))

            loss = 0
            acc = 0
            np.random.shuffle(p)

            pb = tf.contrib.keras.utils.Progbar(n_batch)

            loss_total = 0
            acc_total = 0

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

                if args.dev is not None:
                    X_cv_coo = X_cv.tocoo()
                    X_cv_batch = tf.SparseTensorValue(
                        indices=np.stack([X_cv_coo.row, X_cv_coo.col], axis=1),
                        values=X_cv_coo.data,
                        dense_shape=X_cv_coo.shape
                    )

                    fd_cv = {
                        X_: X_cv_batch,
                        y_: y_cv,
                        w_: w_cv
                    }

                _, loss_batch, acc_batch = sess.run([train_op, loss_op, acc_op], feed_dict=fd)
                pb.update((i/batch_size)+1, values=[('loss', loss_batch), ('acc', acc_batch)])

                loss_total += loss_batch * X_coo.shape[0]
                acc_total += acc_batch * X_coo.shape[0]

            sess.run(incr_global_step)
            saver.save(sess, args.outdir + '/model.ckpt')

            loss_total /= n
            acc_total /= n

            if (iteration-1) % 5 == 0:
                if not args.mlr:
                    dump_embeddings(args.outdir, args.embdim)
                    dump_weights(args.outdir, args.embdim)

            loss_cv, acc_cv = sess.run([loss_op, acc_op], feed_dict=fd_cv)

            summary_iter_train = sess.run(summary, feed_dict = {loss_total_summary:loss_total, acc_total_summary:acc_total})
            summary_iter_cv = sess.run(summary, feed_dict = {loss_total_summary:loss_cv, acc_total_summary:acc_cv})
            writer_train.add_summary(summary_iter_train, global_step.eval(session=sess))
            writer_cv.add_summary(summary_iter_cv, global_step.eval(session=sess))

            if args.nexamples > 0:
                ix = np.random.choice(X.shape[0], args.nexamples)
                X_coo = X[ix].tocoo()
                X_dense = X[ix].todense()
                X_example = tf.SparseTensorValue(
                    indices=np.stack([X_coo.row, X_coo.col], axis=1),
                    values=X_coo.data,
                    dense_shape=X_coo.shape
                )
                fd_example = {
                    X_: X_example,
                    y_: y[ix],
                    w_: w[ix]
                }
                preds, acc = sess.run([hard_predictions, acc_op], feed_dict=fd_example)

                sys.stderr.write('Output examples:\n')

                for i in range(len(preds)):
                    feat_ix = np.where(X_dense[i])[1]
                    feats = []
                    for f in feat_ix:
                        feats.append(d.F.ix2condition[f])

                    sys.stderr.write('  Source Index:  %s\n' %(index[ix[i]]))
                    sys.stderr.write('  Feats:         %s\n' %(','.join(feats)))
                    sys.stderr.write('  Pred:          %s\n' %(d.F.ix2consequent[preds[i]]))
                    sys.stderr.write('  Gold:          %s\n' %(d.F.ix2consequent[y[ix[i]]]))
                    sys.stderr.write('  Weight:        %s\n' %(w[ix[i]]))
                    sys.stderr.write()

                sys.stderr.write('Sample accuracy: %s\n' %acc)
                sys.stderr.write()


        if not args.mlr:
            sys.stderr.write('Linearizing weight matrices...\n')
            dump_embeddings(args.outdir, args.embdim)
            dump_weights(args.outdir, args.embdim)


