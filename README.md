# DNN-DST
Dialog State Tracking with Deep Neural Networks

This is work for my Master's thesis using Deep Belief Networks for Dialogue State Tracking.  You can read more about Dialog State Tracking at http://research.microsoft.com/en-us/events/dstc/

Vectorize.py extracts the relevant information from the log files given in the training sets for the DSTC and converts them into a set of feature vectors.  The features that we used are based off of the features used by Henderson et al in their paper 'Deep Neural Network Approach for the Dialogue State Tracking Challenge' http://mi.eng.cam.ac.uk/~mh521/papers/Deep_Neural_Network_for_Dialog_State_Tracking.pdf

StackedRBM.py is the implementation of our Deep Belief Net based off of the implementation of the Deep Belief Net from the LISA lab at the University of Montreal https://github.com/lisa-lab/DeepLearningTutorials.  The original implementation was designed for use with the MNIST data set so some functions for loading our feature vectors properly were added and the pretraining/training was changed to accomidate for the new dataset.
