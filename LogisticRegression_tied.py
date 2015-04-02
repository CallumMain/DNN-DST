import numpy

import theano
import theano.tensor as T

class LogisticRegression_tied(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, X, n_in):
        """ Initialize the parameters of the logistic regression

        :type input: [theano.tensor.fmatrix]
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)
	:type idx: theano.tensor.TensorType
        :param part: number of rows for each input examples

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        """

        # initialize with 0 the weights w as a vector of dimension n_in
        self.w = theano.shared(value=numpy.zeros(n_in,dtype=theano.config.floatX), 
                                name='w', borrow=True)

        self.Xw = T.dot(X,self.w)    

        # parameters of the model
        
        self.params = [self.w]

    def negative_log_likelihood(self, Y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: [start_pos,end_pos,label]

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
	
	#log_p_y_given_x = T.log(T.nnet.softmax(self.Xw[0:2])[0])
        log_p_y_given_x, _ = theano.scan(fn=lambda y: T.log(T.exp(self.Xw[y[0]+y[2]])/T.sum(T.exp(self.Xw[y[0]:y[1]]))), #T.log(T.nnet.softmax(self.Xw[y[0]:y[1]])[0]), 
                              outputs_info=None,
                              sequences=Y)
	
	return  -T.mean(log_p_y_given_x)
	

    def errors(self, Y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: [start_pos,end_pos,label]
        """

	
	y_err, _ = theano.scan(fn=lambda y: T.neq(T.argmax(self.Xw[y[0]:y[1]]),y[2]),
                              outputs_info=None,
                              sequences=Y)
	
	#y_err = T.neq(T.argmax(self.Xw[0:2]),0)

    	return T.mean(y_err)
        
