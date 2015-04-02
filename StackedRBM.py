"""
"""
import cPickle
import gzip
import os
import sys
import time
import json
import pprint
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

#from logistic_sgd import LogisticRegression, load_data
from LogisticRegression_tied import LogisticRegression_tied
from mlp import HiddenLayer
from rbm import RBM
from HypoIndexer import HypoIndexer


class DBN(object):
    """Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=46*4,
                 hidden_layers_sizes=[20, 20], n_outs=10):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.imatrix('y')  # the labels are presented as 1D vector
                                 # of [int] labels

        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # its arguably a philosophical question...  but we are
            # going to only declare that the parameters of the
            # sigmoid_layers are parameters of the DBN. The visible
            # biases in the RBM are parameters of those RBMs, but not
            # of the DBN.
            self.params.extend(sigmoid_layer.params)

            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression_tied(
            X=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1]) 
            #, n_out=n_outs)
        self.params.extend(self.logLayer.params)

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size, k):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size
	#print batch_begin, batch_end, batch_size, k
        pretrain_fns = []
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=None, k=k)

            # compile the theano function
            fn = theano.function(inputs=[index,
                            theano.Param(learning_rate, default=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x:
                                    train_set_x[batch_begin:batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                        the has to contain three pairs, `train`,
                        `valid`, `test` in this order, where each pair
                        is formed of two Theano variables, one for the
                        datapoints, the other for the labels
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage

        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
	# n_train_size = train_set_y.get_value(borrow=True).shape[0]
      
        n_valid_batches = valid_set_y.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_y.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.iscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

	#shift the indeces in Y for each mini batch
	offset = theano.shared(value=numpy.asarray([[1,1,0] for i in range(batch_size)], dtype='int32'),
				name='offset')

        train_fn = theano.function(inputs=[index], on_unused_input='ignore',
              outputs=self.finetune_cost,
              updates=updates,
              givens={self.x: train_set_x[ train_set_y[index * batch_size][0]:train_set_y[(index+1) * batch_size][0] ], 
					  #index * batch_size:
                                          #(index + 1) * batch_size],
                      self.y: train_set_y[index * batch_size:
                                          (index + 1) * batch_size]-offset*train_set_y[index * batch_size][0]})

        test_score_i = theano.function([index], self.errors, on_unused_input='ignore',
                 givens={self.x: test_set_x[ test_set_y[index * batch_size][0]:test_set_y[(index+1) * batch_size][0] ],
					    #index * batch_size:
                                            #(index + 1) * batch_size],
                         self.y: test_set_y[index * batch_size:
                                            (index + 1) * batch_size]-offset*test_set_y[index * batch_size][0]})

        valid_score_i = theano.function([index], self.errors, on_unused_input='ignore',
              givens={self.x: valid_set_x[ valid_set_y[index * batch_size][0]:valid_set_y[(index+1) * batch_size][0] ],
					  #index * batch_size:
                                          #(index + 1) * batch_size],
                      self.y: valid_set_y[index * batch_size:
                                          (index + 1) * batch_size]-offset*valid_set_y[index * batch_size][0]})

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches-1)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches-1)]

        return train_fn, valid_score, test_score

def LoadData(dataset, window_size=3, dim = 46):

    json_data = open(dataset)

    data = json.load(json_data)

    DX = {}
    DY = {}
    DIM = dim*(window_size+1)

    for session in data["sessions"]:
	X = []
	Y = []
	indexer = HypoIndexer()
        for turn in session["turns"]:

            for slot in turn["x"].keys():
		for value in turn["x"][slot].keys():
		    indexer.getIndex(slot,value)
	    #vectors for current turn
	    x = {}
	    y = {}
	    #initialize the size 
	    for slot in indexer.getSlots():
		x[slot] = [[0]*dim for i in range(indexer.getHypoNum(slot))]
		if "y" in turn.keys():
		    y[slot] = [0,0,-1]

	    for slot in turn["x"].keys():
                for value in turn["x"][slot].keys():
		    vector = []
                    for feat in turn["x"][slot][value].values():
                        vector.append(feat)

		    i = indexer.getIndex(slot,value)

		    if len(vector) != dim :
			print 'Error: dimesion mismatch'

                    x[slot][i] = vector

	    if "y" in turn.keys():
		for slot in turn["y"].keys():
		    if slot in y.keys() :
		        j = indexer.getIndex(slot,turn["y"][slot],False)
		        y[slot][-1] = j
	    X.append(x)
	    Y.append(y)

	for i in range(len(session["turns"])):
	    for slot in X[i].keys():
		if slot not in DX.keys():
		    DX[slot] = []
		    DY[slot] = []

		#always insert a zero vector as bias
		DX[slot].append([0]*DIM)

		for j in range(len(X[i][slot])):
		    phi = [0]*DIM
		    #features in the window
		    for k in range(min(window_size,i)):
			if (slot not in X[i-k].keys()) or (j >= len(X[i-k][slot])) :
			    continue
			phi[k*dim:(k+1)*dim] = X[i-k][slot][j]
		    #summarized features (out of the window)
		    for k in range(max(i-window_size+1,0)):
			if (slot not in X[k].keys()) or (j >= len(X[k][slot])) :
			    continue
			for m in range(dim) :
			    phi[-dim+m] = phi[-dim+m] + X[k][slot][j][m]
		    #add to data set
		    DX[slot].append(phi)
		if not Y[i] :
		    continue
		y = Y[i][slot]
		y[0] = len(DX[slot])-len(X[i][slot])-1
		y[1] = len(DX[slot])
		y[2] = y[2] + 1
		DY[slot].append(y)
		

    
    json_data.close()

    
 #   for slot in DX.keys():
 #	DX[slot] = numpy.array(DX[slot],dtype=theano.config.floatX)
 #	DY[slot] = numpy.array(DY[slot],dtype=int32)

    return DX,DY

def check_alignment(DX,DY): #for debug
    for slot in DX.keys() :
	if len(DX[slot]) != DY[slot][-1][1] :
	    print slot, len(DX[slot]), DY[slot][-1]
    
    for slot in DY.keys() :
	for i in range(1,len(DY[slot])) :
	    if DY[slot][i][0] == DY[slot][i][1] :
		print slot, i, DY[slot][i]
	    if DY[slot][i][0] != DY[slot][i-1][1] :
		print slot, i, DY[slot][i]
	    if DY[slot][i][2] >= DY[slot][i][1]-DY[slot][i][0]:
		print slot, i, DY[slot][i]
		
def join_all(dataset_list, adjust=False):

    train_set = []
    for D in dataset_list :
	for slot in D.keys() :
	    for d in D[slot] :
		train_set.append(d)

    #adjust indeces for Y
    if len(train_set[0]) == 3 and adjust:
	for i in range(len(train_set)) :
	    if i == 0 :
		continue
            delta = train_set[i][1]-train_set[i][0]
	    train_set[i][0] = train_set[i-1][1]
	    train_set[i][1] = train_set[i][0]+delta
    
    return train_set

def train_StackedRBM(finetune_lr=0.1, pretraining_epochs=10,
             pretrain_lr=0.01, k=1, finetuning_epochs=10,
             labelled_dataset=['train1a.half1.json','train1a.half2.json','test4.json'],
	     unlabelled_dataset=['train1b.json'], batch_size=20):
    """
    Demonstrates how to train and test a Deep Belief Network.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining
    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training
    :type k: int
    :param k: number of Gibbs steps in CD/PCD
    :type training_epochs: int
    :param training_epochs: maximal number of iterations ot run the optimizer
    :type dataset: string
    :param dataset: path the the pickled dataset
    :type batch_size: int
    :param batch_size: the size of a minibatch
    """

    dim = 46
    window_size = 3

    train_set_x, train_set_y = LoadData(labelled_dataset[0],window_size,dim)
    valid_set_x, valid_set_y = LoadData(labelled_dataset[1],window_size,dim)
    test_set_x, test_set_y = LoadData(labelled_dataset[2],window_size,dim)
	
    check_alignment(train_set_x, train_set_y)
    check_alignment(valid_set_x, valid_set_y)
    check_alignment(test_set_x, test_set_y)

    all_train_x = [train_set_x]
    for filename in unlabelled_dataset:
    	u_train_set_x, _ = LoadData(filename,window_size,dim)
    all_train_x.append(u_train_set_x)

    shared_train_x = theano.shared(numpy.asarray(join_all(all_train_x), 
                                 	dtype=theano.config.floatX),
                                 	borrow=True)

    #train_set_x, train_set_y = datasets[0]
    #valid_set_x, valid_set_y = datasets[1]
    #test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = shared_train_x.get_value(borrow=True).shape[0] / batch_size


    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print '... building the model'
    # construct the Deep Belief Network
    dbn = DBN(numpy_rng=numpy_rng, n_ins=dim*(window_size+1),
              hidden_layers_sizes=[20, 20, 20],
              n_outs=10)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = dbn.pretraining_functions(train_set_x=shared_train_x,
                                                batch_size=batch_size,
                                                k=k)

    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    for i in xrange(dbn.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)

    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ##################
    # FINETUNE MODEL #
    ##################
    for slot in train_set_x.keys() :

	if (slot in valid_set_x.keys()) and (slot in test_set_x.keys()) :
	    shared_finetune_x = theano.shared(numpy.asarray(train_set_x[slot], 
                                 dtype=theano.config.floatX),
                                 borrow=True)
	    shared_finetune_y = theano.shared(numpy.asarray(train_set_y[slot], 
                                 dtype='int32'),
                                 borrow=True)
	    
	    shared_valid_x = theano.shared(numpy.asarray(valid_set_x[slot], 
                                 dtype=theano.config.floatX),
                                 borrow=True)
	    shared_valid_y = theano.shared(numpy.asarray(valid_set_y[slot], 
                                 dtype='int32'),
                                 borrow=True)

	    shared_test_x = theano.shared(numpy.asarray(test_set_x[slot], 
                                 dtype=theano.config.floatX),
                                 borrow=True)
	    shared_test_y = theano.shared(numpy.asarray(test_set_y[slot], 
                                 dtype='int32'),
                                 borrow=True)

	    #if valid set not available, use all slots to train
	elif (slot not in valid_set_x.keys()) and (slot in test_set_x.keys()) :
	    print 'no validation set available for slot '+slot+'...using all slots to train...'
            shared_finetune_x = theano.shared(numpy.asarray(join_all([train_set_x]), 
                                 	dtype=theano.config.floatX),
                                 	borrow=True)
	    shared_finetune_y = theano.shared(numpy.asarray(join_all([train_set_y],True), 
                                 	dtype='int32'),
                                 	borrow=True)
	    
	    shared_valid_x = theano.shared(numpy.asarray(join_all([valid_set_x]), 
                                 	dtype=theano.config.floatX),
                                 	borrow=True)
	    shared_valid_y = theano.shared(numpy.asarray(join_all([valid_set_y],True), 
                                 	dtype='int32'),
                                 	borrow=True)

	    shared_test_x = theano.shared(numpy.asarray(test_set_x[slot], 
                                	dtype=theano.config.floatX),
                                	borrow=True)
	    shared_test_y = theano.shared(numpy.asarray(test_set_y[slot], 
                                	dtype='int32'),
                                	borrow=True)
	else :
	    print 'no test set available for slot '+slot+'...ignored!...'
	    continue
		 

	datasets = [(shared_finetune_x,shared_finetune_y),(shared_valid_x,shared_valid_y),(shared_test_x,shared_test_y)]	   
	   

	(train_model,validate_model,test_model) =  dbn.build_finetune_functions(datasets=datasets, 
								batch_size=batch_size, 
								learning_rate=finetune_lr)

	n_train_batches = shared_finetune_y.get_value(borrow=True).shape[0] / batch_size-1

	print '... fine-tuning: '+slot

	# early-stopping parameters
	patience = 1000  # look as this many examples regardless
	patience_increase = 2  # wait this much longer when a new best is
		                   # found
	improvement_threshold = 0.995  # a relative improvement of this much is
		                           # considered significant
	validation_frequency = min(n_train_batches, patience / 2)
		                          # go through this many
		                          # minibatche before checking the network
		                          # on the validation set; in this case we
		                          # check every epoch

	best_params = None
	best_validation_loss = numpy.inf
	best_iter = 0
	test_score = 0.
	start_time = time.clock()

	epoch = 0
	done_looping = False

	while (epoch < finetuning_epochs) and (not done_looping):
	    epoch = epoch + 1
	    for minibatch_index in xrange(n_train_batches):

		minibatch_avg_cost = train_model(minibatch_index)
		# iteration number
		iter = (epoch - 1) * n_train_batches + minibatch_index

		if (iter + 1) % validation_frequency == 0:
		    # compute zero-one loss on validation set
		    validation_losses = validate_model() #[validate_model(i) for i
		                             		#in xrange(n_valid_batches)]
		    this_validation_loss = numpy.mean(validation_losses)

		    print('epoch %i, minibatch %i/%i, validation error %f %%' %
		             (epoch, minibatch_index + 1, n_train_batches,
		              this_validation_loss * 100.))

		    # if we got the best validation score until now
		    if this_validation_loss < best_validation_loss:
		        #improve patience if loss improvement is good enough
		        if this_validation_loss < best_validation_loss *  \
		                   improvement_threshold:
		            patience = max(patience, iter * patience_increase)

		        best_validation_loss = this_validation_loss
		        best_iter = iter

		        # test it on the test set
		        test_losses = test_model() #[test_model(i) for i
		                           		#in xrange(n_test_batches)]
		        test_score = numpy.mean(test_losses)

		        print(('     epoch %i, minibatch %i/%i, test error of '
		                   'best model %f %%') %
		                  (epoch, minibatch_index + 1, n_train_batches,
		                   test_score * 100.))

		if patience <= iter:
		        done_looping = True
		        break

	end_time = time.clock()

	print(('Optimization complete. Best validation score of %f %% '
		  'obtained at iteration %i, with test performance %f %%') %
		  (best_validation_loss * 100., best_iter + 1, test_score * 100.))
	print >> sys.stderr, ('The code for file ' +
		              os.path.split(__file__)[1] +
		              ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    #theano.config.exception_verbosity = 'high'
    train_StackedRBM()
