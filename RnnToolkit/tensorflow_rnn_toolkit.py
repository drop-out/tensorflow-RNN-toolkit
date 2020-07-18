import tensorflow as tf
import pandas as pd
import numpy as np
from random import shuffle

class BatchGenerator:
    def __init__(self,n_observations):
        self.observation_index = list(range(n_observations))
        self.n_observations = n_observations
        self.pointer = 0
        shuffle(self.observation_index)
        
    def reset_pointer(self):
        self.pointer = 0
        shuffle(self.observation_index)
        
    def next_batch(self,batch_size):
        if self.pointer + batch_size > self.n_observations:
            self.reset_pointer()
        result = self.observation_index[self.pointer:self.pointer+batch_size]
        self.pointer += batch_size
        return result
    
    def test_set(self,batch_size):
        n_test_batch = self.n_observations//batch_size
        result=[]
        for i in range(n_test_batch):
            result.append(list(range(i*batch_size,(i+1)*batch_size)))
        residual = self.n_observations%batch_size
        if residual>0:
            n_test_batch+=1
            result.append(list(range((i+1)*batch_size,(i+1)*batch_size+residual)))
        return n_test_batch,result

def get_sequence_array_from_dataframe(dataframe,individual_name,date_name,feature_list,label_list):
    '''
    This function converts dataframes to tensorflow-type numpy arrays.
    Dataframes should be formatted with columns [individual,date,features...,labels...].
    The returned array is of shape (batch_size,sequence_length,number_of_features or number_of_labels)
    '''
    dataframe['time_step#']=dataframe.groupby([individual_name])[date_name].rank().astype(int).values
    seq_length=dataframe.groupby([individual_name],as_index=False)['time_step#'].max()
    individual_index=seq_length[individual_name].values
    seq_length=seq_length['time_step#'].values
    
    feature=[]
    for i in feature_list:
        feature.append(dataframe.pivot(index=individual_name, columns='time_step#', values=i).values)
    feature=np.stack(feature,axis=-1)
    
    label=[]
    for i in label_list:
        label.append(dataframe.pivot(index=individual_name, columns='time_step#', values=i).values)
    label=np.stack(label,axis=-1)
    
    return individual_index,seq_length,feature,label


def acc(pred,true,mask):
    '''
    This function calculates accuracy.
    '''
    pred=tf.cast(pred,tf.int64)
    true=tf.cast(true,tf.int64)
    equal=tf.cast(tf.equal(pred,true),tf.float32)
    acc=tf.reduce_sum(equal*mask)/tf.reduce_sum(mask)
    return acc

def get_rnn_output(input_seq,seq_length=None,cell=tf.nn.rnn_cell.GRUCell,activation=tf.tanh,n_hidden_units=10,name='RNN1',**kwargs):
    '''
    RNN layer building block.
    ''' 

    batch_size=tf.shape(input_seq)[0]
    
    #RNN layer
    cell=cell(num_units=n_hidden_units,activation=activation,name=name+'_cell',**kwargs)
    
    #initial_state
    #initial_state = cell.zero_state(batch_size, dtype=tf.float32)
    initial_state = tf.get_variable(name+'_initial_state',[1,n_hidden_units])
    initial_state = tf.tile(initial_state, tf.stack([batch_size, 1]))
    
    outputs, state = tf.nn.dynamic_rnn(cell, input_seq,seq_length,
                                       initial_state=initial_state)

    return outputs, state

def get_fc_output(FC_input,activation=tf.nn.leaky_relu,n_hu_in=10,n_hu_out=10,name='FC1',):
    '''
    FC layer building block.
    ''' 
    W=tf.get_variable(name+'_W',[n_hu_in,n_hu_out])
    b=tf.get_variable(name+'_b',[n_hu_out])
    
    FC_output=activation(tf.matmul(FC_input,W)+b)
    
    return FC_output

def get_fc_with_bn_output(FC_input,activation=tf.nn.leaky_relu,target='train',n_hu_in=10,n_hu_out=10,name='FC1'):
    '''
    FC layer building block, with batch normalization.
    ''' 
    W=tf.get_variable(name+'_W',[n_hu_in,n_hu_out])
    b=tf.get_variable(name+'_b',[n_hu_out])
    FC_output=tf.matmul(FC_input,W)+b
    FC_output=tf.layers.batch_normalization(FC_output,name=name+'BN',training=True if target=='train' else False)
    FC_output=activation(FC_output)
    
    return FC_output

def get_fc_output_from_sequence(sequence,activation=tf.nn.leaky_relu,n_input=20,n_output=3,name='task1'):
    '''
    FC layer building block that should be used when input and output are sequence data.
    ''' 
    batch_size=tf.shape(sequence)[0]
    seq_length=tf.shape(sequence)[1]
    
    W_1=tf.get_variable(name+'_W_1',[n_input,n_output])
    b_1=tf.get_variable(name+'_b_1',[n_output])
    
    sequence=tf.reshape(sequence,[-1,n_input])
    sequence=tf.matmul(sequence,W_1)+b_1
    sequence=activation(sequence)

    output=tf.reshape(sequence,tf.stack([batch_size,seq_length,n_output]))
    
    return output

def get_fc_with_bn_output_from_sequence(sequence,activation=tf.nn.leaky_relu,target='train',n_input=3,n_output=3,name='task1'):
    '''
    FC layer building block(with batch normalization) that should be used when input and output are sequence data.
    ''' 
    batch_size=tf.shape(sequence)[0]
    seq_length=tf.shape(sequence)[1]
    
    W_1=tf.get_variable(name+'_W_1',[n_input,n_output])
    b_1=tf.get_variable(name+'_b_1',[n_output])
    
    sequence=tf.reshape(sequence,[-1,n_input])
    sequence=tf.matmul(sequence,W_1)+b_1
    sequence=tf.layers.batch_normalization(sequence,name=name+'BN',training=True if target=='train' else False)
    sequence=activation(sequence)

    output=tf.reshape(sequence,tf.stack([batch_size,seq_length,n_output]))
    
    return output