import pandas as pd
import tensorflow as tf
import numpy as np

feature_list=['f1','f2','f3']
label_list=['l1','l2','l3']

dataframe = pd.read_csv('train_data.csv',parse_dates=['date'])
feed_index,feed_seq_length,feed_feature,feed_label=get_sequence_array_from_dataframe(dataframe,
                                                                                    individual_name='member_id', 
                                                                                    date_name='date',
                                                                                    feature_list=feature_list,
                                                                                    label_list=label_list)

class PARAS:
    seq_length = 100
    n_feature = 3
    n_label = 3
    n_hu = 10
    l2 = 0.01


class Model():
    def __init__(self,target='train'):
        with tf.device('/cpu:0'):
            tf.reset_default_graph()

            #inputs
            lr=tf.placeholder(tf.float32,[],name='learning_rate')
            feature=tf.placeholder(tf.float32,[None,None,PARAS.n_feature])
            label=tf.placeholder(tf.float32,[None,None,PARAS.n_label])


            #RNN layer
            with tf.variable_scope('RNN',reuse=tf.AUTO_REUSE):
                output,state=get_rnn_output(feature,
                                      cell=tf.nn.rnn_cell.GRUCell,
                                      n_hidden_units=PARAS.n_hu,
                                      name='RNN')
                
            #FC layer
            with tf.variable_scope('FC',reuse=tf.AUTO_REUSE):
                logits=get_fc_with_bn_output_from_sequence(output,
                                                           activation=tf.nn.relu,
                                                           target=target,
                                                           n_hidden_units=PARAS.n_hu,
                                                           n_target=PARAS.n_label,
                                                           name='FC')
            
            #output layer
            prob=tf.nn.sigmoid(logits)
            pred=tf.cast(logits>0,tf.int64)

            #loss layer
            regularizer=tf.contrib.layers.l2_regularizer(PARAS.l2)
            penalty=tf.contrib.layers.apply_regularization(regularizer,tf.trainable_variables())

            obj=tf.losses.sigmoid_cross_entropy(label,logits)
            obj_with_penalty=obj+penalty

            train_acc=acc(pred,label,mask=tf.ones_like(label))

            #optimizatin layer
            optimizer=tf.train.AdamOptimizer(lr)
            grads = optimizer.compute_gradients(obj_with_penalty)

            #gradient clipping
            for i, (g, v) in enumerate(grads):
                if g is not None:
                    grads[i] = (tf.clip_by_norm(g, 1), v)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step=optimizer.apply_gradients(grads)

            self.train_step=train_step
            self.obj=obj
            self.obj_with_penalty=obj_with_penalty
            self.train_acc=train_acc
            self.feature=feature
            self.label=label
            self.lr=lr


#Train
model= Model(target='train')
sess=tf.Session()
sess.run(tf.global_variables_initializer())
_,_acc = sess.run([model.train_step,model.train_acc],feed_dict={model.label:feed_label,model.feature:feed_feature,model.lr:0.01})

print(_acc)