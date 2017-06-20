import numpy as np
# theano imports                                                                                        
import theano
import theano.tensor as T

# keras imports                                                                                         
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Input, Activation
from keras.optimizers import SGD, adam
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization

from data_utils import Dataset_iter, DatasetBatchBuffer_iter

import matplotlib as mpl
mpl.use('Agg') # no need for X-server
from matplotlib import pyplot as plt

def plot_feats(generated_feats, epoch, index, ext=''):

    plt.figure()
    for row in generated_feats:
        plt.plot(row)
    plt.savefig('figures/mean_pulse_epoch{}_index{}'.format(epoch, index) + ext + '.png')
    plt.close()


def model(timesteps=128, input_dim=31, output_dim=400, model_name="glot_model"):

    ac_input = Input(shape=(input_dim,), name="ac_input")

    x = ac_input                                                                                       
    
    x = Dense(256, activation='linear', kernel_initializer='glorot_normal')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)                      
    
    x = Dense(512, activation='linear', kernel_initializer='glorot_normal')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)                      
    
    x = Dense(output_dim, activation='linear', kernel_initializer='glorot_normal')(x)               
    
    
    #model = Model(inputs=[ac_input], outputs=[x], name=model_name)
    model = Model(inputs=ac_input, outputs=x, name=model_name)

    return model


def train_model(dataset, batch_size=128,
                input_keys=['f0', 'lsf'], output_keys=['pls']):

    output_dim = dataset.getDim(output_keys)
    input_dim = dataset.getDim(input_keys)
    print output_dim
    print input_dim
    

    optim = adam(lr=0.0001)
    # train glot model in time domain first
 
    no_epochs = 1
    max_epochs_no_improvement = 5

    #dataset.DataProvider('./data_provider.dat')

    mdl = model(input_dim=input_dim, output_dim=output_dim)
    mdl.compile(loss=['mse'], loss_weights=[1.0], optimizer=optim)
    
    loss_test_max = 1e20
    loss_val_max = 1e20
    patience = max_epochs_no_improvement
    for epoch in range(no_epochs):

        print("Training epoch is", epoch)
        total_batches = 0
        val_data = []

        batch_idx = 0
        for data_batch in DatasetBatchBuffer_iter(dataset,
                                                  shuffle=False,
                                                  batch_size=batch_size,
                                                  io_keys=[input_keys, output_keys],
                                                  subset='train'):
            input_batch = data_batch.inputs
            output_batch =  data_batch.outputs
            
            

            
            d = mdl.train_on_batch([input_batch],
                                   [output_batch])
            
            
            
            print("training batch %d, loss: %f" %
                      (batch_idx, d))

            

            batch_idx += 1
            

        """
        data_dict = dataset.splitData(keys=input_keys, data=input_batch)
        plt_ind = np.where(data_dict['f0'] > 0)
        print plt_ind
        print data_dict['f0']
        #print data_dict.keys()
        #print data_dict.values()   
        """
      
        plt_pred = mdl.predict([input_batch], verbose=0)
        plt_pred = plt_pred[0,:]
        plt_tar = output_batch[0,:]
        plt_data = np.array([plt_pred, plt_tar])
        plot_feats(plt_data, epoch, batch_idx, ext='.pulses')

        loss_val = 0.0
        for data in Dataset_iter(dataset, subset='val',
                                 io_keys=[input_keys, output_keys]):
            
            loss_val += mdl.evaluate(data.inputs, data.outputs, verbose=False)
            
        if loss_val < loss_val_max:
            loss_val_max = loss_val
            patience = max_epochs_no_improvement
        else: 
            print("early stopping at epoch: %d"  % (epoch))
            
        loss_test = 0.0    
        for data in Dataset_iter(dataset, subset='test',
                                 io_keys=[input_keys, output_keys]):
            
            loss_test += mdl.evaluate(data.inputs, data.outputs, verbose=False)


        print("validation loss = %f" % (loss_val))
        print("test loss = %f" % (loss_test))

    ##mdl.save_weights('./models/pls.model', overwrite=True)
    
    
def generate(dataset, batch_size=128,
             input_keys=['f0', 'lsf'], output_keys=['pls'],
             out_dir=None):
    
    
    output_dim = dataset.getDim(output_keys)
    input_dim = dataset.getDim(input_keys)
    
    optim = adam(lr=0.0001)
    mdl = model(input_dim=input_dim, output_dim=output_dim)
    mdl.compile(loss=['mse'], loss_weights=[1.0], optimizer=optim)
    mdl.load_weights('./models/pls.model')
    
    
    for data in Dataset_iter(dataset, subset='test',
                             io_keys=[input_keys, output_keys]):

  
        output_pred = mdl.predict(data.inputs)
        # todo: revert normalization of outputs
        
        # todo: write function to split output data streams
        ext = dataset.DataObjs[output_keys[0]].ext
        dtype = dataset.DataObjs[output_keys[0]].dtype
        
        out_file =  out_dir + "/" + data.tag + ext
        
        #print output_pred.shape
        #print out_file
        
        output_pred.astype(dtype).tofile(out_file)
        
    
    
