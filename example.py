import pandas as pd
import tensorflow as tf
import numpy as np
import RnnToolkit as rt

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
                output,state=rt.get_rnn_output(feature,
                                      cell=tf.nn.rnn_cell.GRUCell,
                                      n_hidden_units=PARAS.n_hu,
                                      name='RNN1')
                
                output_temp,state=rt.get_rnn_output(output,
                                      cell=tf.nn.rnn_cell.GRUCell,
                                      n_hidden_units=PARAS.n_hu,
                                      name='RNN2')
                
                output+=output_temp # 残差连接
                
            #FC layer
            with tf.variable_scope('FC',reuse=tf.AUTO_REUSE):
                logits=rt.get_fc_with_bn_output_from_sequence(output,
                                                           activation=tf.nn.relu,
                                                           target=target,
                                                           n_input=PARAS.n_hu,
                                                           n_output=PARAS.n_hu,
                                                           name='FC_1')

                # should not add batch normalization to the last layer
                logits=rt.get_fc_output_from_sequence(logits,
                                                           activation=tf.identity,
                                                           target=target,
                                                           n_input=PARAS.n_hu,
                                                           n_output=PARAS.n_label,
                                                           name='FC_2')
            
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

            self.train_step = train_step
            self.obj = obj
            self.obj_with_penalty = obj_with_penalty
            self.train_acc = train_acc
            self.feature = feature
            self.label = label
            self.pred = pred
            self.lr = lr


# Train
model= Model(target='train')
model_name = 'model_train'
global_step = 0
cum_accuracy=[0.0]*100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables()) # 模型保存器(checkpoint)
    if global_step>0:
        saver.restore(sess,'out/model/%s.ckpt-%s'%(model_name,global_step-1))
    for i in range(1,10001):
        _,_acc = sess.run([model.train_step,model.train_acc],feed_dict={model.label:feed_label,model.feature:feed_feature,model.lr:0.01})
        cum_accuracy = cum_accuracy[1:]+[_acc]
        if i%100==0:
            print(sum(cum_accuracy)/100)
        if i%1000==0:
            saver.save(sess,'out/model/%s.ckpt'%model_name, global_step=global_step)
            global_step+=1
    
# Predict
model= Model(target='predict')
model_name = 'model_train'
global_step = 11
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables()) # 模型保存器(checkpoint)
    saver.restore(sess,'out/model/%s.ckpt-%s'%(model_name,global_step-1))
    _pred = sess.run(model.pred,feed_dict={model.feature:feed_feature})
    
# Saved Model
model= Model(target='predict')
model_name = 'model_train'
global_step = 11
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables()) # 模型保存器(checkpoint)
    saver.restore(sess,'out/model/%s.ckpt-%s'%(model_name,global_step-1))
    saver_savedmodel = tf.saved_model.simple_save(sess,
            "out/model/saved_model",
            inputs={"feature":model.feed_feature},
            outputs={"pred":model.pred})
    
# Predict from saved_model
predictor = tf.contrib.predictor.from_saved_model("out/model/saved_model")
predictor({"feature":feed_feature})
