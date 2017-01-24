import os
import sys
import timeit

import numpy
import scipy.io as sio

import theano
import theano.tensor as T
#from theano.tensor.signal import downsample
#from theano.tensor.nnet import conv 

# Config file 
import imp # for importing argv[1]
if len(sys.argv) < 2:
    sys.exit("Usage: python GlottDnnScript.py config.py")
if os.path.isfile(sys.argv[1]):
    conf = imp.load_source('', sys.argv[1])
else:
    sys.exit("Config file " + sys.argv[1] + " does not exist")


from dnnClasses import HiddenLayer, LogisticRegression

def load_data(filename, size2):
    var = numpy.fromfile(filename, dtype=numpy.float32, count=-1, sep='')
    var = numpy.reshape(var,(-1,size2))

    shared_var = theano.shared(numpy.asarray(var, dtype=theano.config.floatX), borrow=True)
    return shared_var
# EOF load_data_mat()

def save_network(layerList, layer_out):
    fid = open( conf.weights_data_dir + '/' + conf.dnn_name + '.dnnData','w')
    for layer in layerList:
        layer.W.get_value().astype(numpy.float32).tofile(fid, sep='',format="%f")
        layer.b.get_value().astype(numpy.float32).tofile(fid, sep='',format="%f")

    layer_out.W.get_value().astype(numpy.float32).tofile(fid, sep='',format="%f")
    layer_out.b.get_value().astype(numpy.float32).tofile(fid, sep='',format="%f")
# EOF : save_network


def list_dir_fullpath(dirname,start,extension):
    output = []
    for f in os.listdir(dirname):
        if f.endswith(extension) & f.startswith(start):
            output.append(os.path.join(dirname,f))
    return output


def evaluate_dnn(learning_rate=0.1, n_epochs=150000,
                    n_in=42, n_out=500,  n_hidden=[100, 250, 500], batch_size=32):
    """ 

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    num_hidden = len(n_hidden)
    
    nndata_basename = conf.train_data_dir + '/' + conf.dnn_name 

    #valid_set_x = load_data(nndata_basename + '.' + str(conf.val_set[0]) + '.idat', n_in)
    #valid_set_y = load_data(nndata_basename + '.' + str(conf.val_set[0]) + '.odat', n_out)
    
    #test_set_x = load_data(nndata_basename + '.' + str(conf.test_set[0]) + '.idat', n_in)
    #test_set_y = load_data(nndata_basename + '.' + str(conf.test_set[0]) + '.odat', n_out)
    
    #train_set_x = load_data(nndata_basename + '.' + str(conf.train_set[0]) + '.idat', n_in)
    #train_set_y = load_data(nndata_basename + '.' + str(conf.train_set[0]) + '.odat', n_out)
    
    valid_set_x = load_data(nndata_basename + '.val.idat', n_in)
    valid_set_y = load_data(nndata_basename + '.val.odat', n_out)
    
    test_set_x = load_data(nndata_basename + '.test.idat', n_in)
    test_set_y = load_data(nndata_basename + '.test.odat', n_out)
    
    train_set_x = load_data(nndata_basename + '.train.idat', n_in)
    train_set_y = load_data(nndata_basename + '.train.odat', n_out)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.matrix('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    print 'Number of hidden layers: %i, Hidden units per layer: i' %(num_hidden)

    layer0_input = x
	
    layerList = []
    for i in xrange(0, num_hidden):
        if i > 0:
            cInput = layerList[i-1].output
            cn_in = n_hidden[i-1]
            cn_out = n_hidden[i]
        else:
            cInput = layer0_input
            cn_in = n_in
            cn_out = n_hidden[0]
        print 'in: %d, out: %d' %(cn_in, cn_out)    
        cLayer = HiddenLayer(
            rng=rng,
            input=cInput,
            n_in=cn_in,
            n_out=cn_out,
            activation = T.nnet.sigmoid
            #activation=relu
        )
        layerList.append(cLayer)
    	
    layer_out = LogisticRegression(input=layerList[-1].output, n_in=n_hidden[-1], n_out=n_out)

    # the cost we minimize during training is the NLL of the model
    cost = layer_out.mse(y,n_out)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer_out.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer_out.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )



    # create a list of all model parameters to be fit by gradient descent
    # params = layer2.params + layer1.params + layer0.params
    params = []
    for layer in layerList:
        params = params + layer.params
    params = params + layer_out.params    

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
	[index],
	cost,
	updates=updates,
	givens={
	    x: train_set_x[index * batch_size: (index + 1) * batch_size],
	    y: train_set_y[index * batch_size: (index + 1) * batch_size]
	}
    )
    

    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    #patience = 10000  # look as this many examples regardless
    patience = conf.patience
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
                    
        # loop over minibatches
        for minibatch_index in xrange(n_train_batches):
            
            iter = (epoch - 1) * n_train_batches + minibatch_index            		
            cost_ij = train_model(minibatch_index)
            
            if (iter + 1) % validation_frequency == 0:
                
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
		
                training_losses = [train_model(i) for i
                                   in xrange(n_train_batches)]
                this_training_loss = numpy.log(numpy.mean(training_losses))
		
                print('epoch %i, minibatch %i/%i, training error %f, validation error %f' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_training_loss, this_validation_loss))
                
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                        
                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                        
                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                        ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score))
                    save_network(layerList, layer_out)
                    #ENDIF (validation_loss is improved)
                #ENDIF (validate at this iteration)
            #ENDFOR (loop minibatches)
            
        if patience <= iter:
            done_looping = True
            break	
        #END WHILE (training epochs)

    end_time = timeit.default_timer()
    print('Optimization complete.')
    #print('Best validation score of %f obtained at iteration %i, '
    #      'with test performance %f' %
    #      (best_validation_loss., best_iter + 1, test_score.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

# EOF evaluate_dnn


if __name__ == '__main__':
   
    dim_in = sum(conf.input_dims)
    dim_out = sum(conf.output_dims)

    evaluate_dnn(n_in=dim_in, n_out=dim_out, n_hidden=conf.n_hidden, batch_size=conf.batch_size, 
                 learning_rate=conf.learning_rate, n_epochs = conf.max_epochs)
