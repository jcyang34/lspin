import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import pairwise_distances as distance_np

def convertToOneHot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)


# Dataset class: adapted from TensorFlow source example:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)


# add a second data label: meta (Z), with the same usage as labels (y)
class DataSet_meta:
    """Base data set class
    """

    def __init__(self, shuffle=True, labeled=True, **data_dict):
        assert '_data' in data_dict
        if labeled:
            assert '_labels' in data_dict
            assert data_dict['_data'].shape[0] == data_dict['_labels'].shape[0]
            assert data_dict['_data'].shape[0] == data_dict['_meta'].shape[0]
        self._labeled = labeled
        self._shuffle = shuffle
        self.__dict__.update(data_dict)
        self._num_samples = self._data.shape[0]
        self._index_in_epoch = 0
        if self._shuffle:
            self._shuffle_data()

    def __len__(self):
        return len(self._data)

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def meta(self):
        return self._meta
    
    @property
    def labeled(self):
        return self._labeled

    @property
    def valid_data(self):
        return self._valid_data

    @property
    def valid_labels(self):
        return self._valid_labels

    @property
    def valid_meta(self):
        return self._valid_meta
    
    @property
    def test_data(self):
        return self._test_data

    @property
    def test_labels(self):
        return self._test_labels

    @property
    def test_meta(self):
        return self._test_meta
    
    @classmethod
    def load(cls, filename):
        data_dict = np.load(filename)
        return cls(**data_dict)

    def save(self, filename):
        data_dict = self.__dict__
        np.savez_compressed(filename, **data_dict)

    def _shuffle_data(self):
        shuffled_idx = np.arange(self._num_samples)
        np.random.shuffle(shuffled_idx)
        self._data = self._data[shuffled_idx]
        if self._labeled:
            self._labels = self._labels[shuffled_idx]
            self._meta = self._meta[shuffled_idx]
            
    def next_batch(self, batch_size):
        assert batch_size <= self._num_samples
        start = self._index_in_epoch
        if start + batch_size > self._num_samples:
            data_batch = self._data[start:]
            if self._labeled:
                labels_batch = self._labels[start:]
                meta_batch = self._meta[start:]
            remaining = batch_size - (self._num_samples - start)
            if self._shuffle:
                self._shuffle_data()
            start = 0
            data_batch = np.concatenate([data_batch, self._data[:remaining]],
                                        axis=0)
            if self._labeled:
                labels_batch = np.concatenate([labels_batch,
                                               self._labels[:remaining]],
                                              axis=0)
                meta_batch = np.concatenate([meta_batch,
                                             self._meta[:remaining]],
                                            axis=0)
            self._index_in_epoch = remaining
        else:
            data_batch = self._data[start:start + batch_size]
            if self._labeled:
                labels_batch = self._labels[start:start + batch_size]
                meta_batch = self._meta[start:start + batch_size]
            self._index_in_epoch = start + batch_size
        batch = (data_batch, labels_batch,meta_batch) if self._labeled else data_batch
        return batch



def squared_distance(X):
    '''
    Calculates the squared Euclidean distance matrix.
    X:              an n-by-p matrix, which includes n samples in dimension p
    returns:        n x n pairwise squared Euclidean distance matrix
    '''

    r = tf.reduce_sum(X*X, 1)
    r = tf.reshape(r, [-1, 1])
    D = r - 2*tf.matmul(X, X, transpose_b=True) + tf.transpose(r)
    return D


class Model(object):
    def __init__(self, 
                 input_node,
                 hidden_layers_node,
                 output_node,
                 gating_net_hidden_layers_node,
                 display_step, 
                 activation_gating,
                 activation_pred,
                 feature_selection=True,
                 batch_normalization=True,
                 a = 1,
                 sigma = 0.5,
                 lam = 0.5, 
                 gamma1 = 0, 
                 gamma2 = 0, 
                 num_meta_label=None,
                 stddev_input=0.1,
                 stddev_input_gates = 0.1,
                 seed=1,
        ): 
        """ LSPIN Model
        # Arguments:
            input_node: integer, input dimension of the prediction network
            hidden_layers_node: list, number of nodes for each hidden layer for the prediction net, example: [200,200]
            output_node: integer, number of nodes for the output layer of the prediction net, 1 (regression) or 2 (classification)
            gating_net_hidden_layers_node: list, number of nodes for each hidden layer of the gating net, example: [200,200]
            display_step: integer, number of epochs to output info
            activation_gating: string, activation function of the gating net: 'relu', 'l_relu', 'sigmoid', 'tanh', or 'none'
            activation_pred: string, activation function of the prediction net: 'relu', 'l_relu', 'sigmoid', 'tanh', or 'none'
            feature_selection: bool, if using the gating net
            a: float, 
            sigma: float, std of the gaussion reparameterization noise
            lam: float, regularization parameter (lambda_1) of the L0 penalty 
            gamma1: float, regularization parameter (lambda_2) to encourage similar samples select similar features 
            gamma2: float, variant of lambda2, to encourage disimilar samples select disimilar features
            num_meta_label: integer, the number of group labels when computing the second regularization term
            stddev_input: float, std of the normal initializer for the prediction network weights
            stddev_input_gates: float, std of the normal initializer for the gating network weights
            seed: integer, random seed
        """

        # Register hyperparameters of LSPIN
        self.a = a
        self.sigma = sigma
        self.lam = lam
        
        self._activation_gating = activation_gating
        self.activation_gating = activation_gating # will overwrite _activation_gating
        
        self._activation_pred = activation_pred
        self.activation_pred = activation_pred # will overwrite _activation_pred
        
        
        # Register hyperparameters for training
        
        #self._batch_size = batch_size
        self.display_step = display_step

        # 2nd regularization parameter for the similarity penalty
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        
        
        G = tf.Graph()
        with G.as_default():
            self.sess = tf.Session(graph=G)
            # tf Graph Input
            X = tf.placeholder(tf.float32, [None, input_node]) # X.shape == [batch_size, feature_size]
            y = tf.placeholder(tf.float32, [None, output_node])
           
            train_gates = tf.placeholder(tf.float32,[1], name='train_gates')
            
            
            # add a second layer of labels to apply penalty
            Z = tf.placeholder(tf.float32, [None, num_meta_label])
            
            is_train = tf.placeholder(tf.bool,[], name='is_train') # for batch normalization
            
            self.learning_rate= tf.placeholder(tf.float32,shape=[], name='learning_rate')
            
            self.compute_sim = tf.placeholder(tf.bool,[], name='compute_sim') # whether to compute the 2nd penalty term or not
            
            self.gatesweights=[]
            nnweights = []
            prev_node = input_node
            prev_x = X
            
            # Gating network
            if feature_selection:
                for i in range(len(gating_net_hidden_layers_node)):
                    gates_layer_name = 'gate_layer' + str(i+1)
                    
                    with tf.variable_scope(gates_layer_name, reuse=tf.AUTO_REUSE):
                        weights = tf.get_variable('weights', [prev_node, gating_net_hidden_layers_node[i]],
                                                  initializer=tf.truncated_normal_initializer(stddev=stddev_input_gates))
                        biases = tf.get_variable('biases', [gating_net_hidden_layers_node[i]],
                                                 initializer=tf.constant_initializer(0.0))
                    
                        self.gatesweights.append(weights)
                        self.gatesweights.append(biases)
                        
                        gates_layer_out = self.activation_gating(tf.matmul(prev_x,weights)+biases)

                        prev_node = gating_net_hidden_layers_node[i]
                        prev_x = gates_layer_out        
                weights2 = tf.get_variable('weights2', [prev_node,input_node],
                                                  initializer=tf.truncated_normal_initializer(stddev=stddev_input_gates))
                biases2 = tf.get_variable('biases2', [input_node],
                                                 initializer=tf.constant_initializer(0.0))
  
                self.gatesweights.append(weights2)
                self.gatesweights.append(biases2)
                self.alpha= self.activation_gating(tf.matmul(prev_x,weights2)+biases2)
                prev_x = X
                self.stochastic_gates = self.get_stochastic_gate_train(prev_x, train_gates)
                prev_x = prev_x * self.stochastic_gates
                #prev_x = self.feature_selector(prev_x, train_gates)
                prev_node = input_node

            # Prediction network
            layer_name = 'layer' + str(1)
            for i in range(len(hidden_layers_node)):
                layer_name = 'layer' + str(i+1)
                with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                    weights = tf.get_variable('weights', [prev_node, hidden_layers_node[i]],
                                              initializer=tf.truncated_normal_initializer(stddev=stddev_input))
                    nnweights.append(weights)
                    biases = tf.get_variable('biases', [hidden_layers_node[i]],
                                             initializer=tf.constant_initializer(0.0))
                    nnweights.append(biases)
                    if batch_normalization:
                        prev_x = tf.layers.batch_normalization(prev_x, training=is_train)
                    layer_out = self.activation_pred(tf.matmul(prev_x, weights) + biases)
               
                    prev_node = hidden_layers_node[i]
                    prev_x = layer_out

            # Output of model
            # Minimize error using cross entropy
            if output_node==1:
                
                weights = tf.get_variable('weights', [prev_node, 1],
                                              initializer=tf.truncated_normal_initializer(stddev=stddev_input))
                nnweights.append(weights)
                biases = tf.get_variable('biases', [1],
                                         initializer=tf.constant_initializer(0.0))
                nnweights.append(biases)
                if batch_normalization:
                    layer_out = tf.layers.batch_normalization(layer_out, training=is_train)
                pred = (tf.matmul(layer_out, weights) + biases)
                loss_fun = tf.reduce_mean(tf.squared_difference(pred, y))
                pred_log = (layer_out)
            else:
                
                weights = tf.get_variable('weights', [prev_node, output_node],
                                              initializer=tf.truncated_normal_initializer(stddev=stddev_input))
                nnweights.append(weights)
                biases = tf.get_variable('biases', [output_node],
                                             initializer=tf.constant_initializer(0.0))
                nnweights.append(biases)
                
                if batch_normalization:
                    prev_x = tf.layers.batch_normalization(prev_x, training=is_train)
                layer_out = self.activation_pred(tf.matmul(prev_x, weights) + biases)
                
                prev_node = output_node
                prev_x = layer_out
                
                pred = tf.nn.softmax(layer_out)
                pred_log = (layer_out)
                loss_fun = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=layer_out))
            
            if feature_selection:
                # gates regularization
                input2cdf = self.alpha
       
                reg = 0.5 - 0.5*tf.erf((-1/(2*self.a) - input2cdf)/(self.sigma*np.sqrt(2)))
                
                reg_gates = self.lam*tf.reduce_mean(tf.reduce_mean(reg,axis=-1))
            
                # 2nd regularization penalty
                # to force samples of the same labels to select similar features
                
                # New feature: reg_sim is only open when the compute_sim flag is true
                #if self.compute_sim:
                    # squared distance between the labels can be either 0 or 2,
                    # scale the corresponding affinity (K_batch) to 1 or 0. 
                    #K_batch = 1.0 - squared_distance(Z)/2.0
                
                    # euclidean distance matrix of the gate matrix
                    #D_batch = squared_distance(self.stochastic_gates)
                
                    # element-wise multiplication
                    #reg_sim = self.gamma1*tf.reduce_mean(tf.reduce_mean(tf.multiply(K_batch,D_batch),axis=-1))+ \
                    #          self.gamma2*tf.reduce_mean(tf.reduce_mean(tf.multiply((1.0 - K_batch),-D_batch)))
                    
                #else:
                    #reg_sim = tf.constant(0.)
                
                reg_sim = tf.cond(self.compute_sim,
                                  lambda: self.gamma1*tf.reduce_mean(tf.reduce_mean(tf.multiply((1.0 - squared_distance(Z)/2.0),                                                                                              squared_distance(self.stochastic_gates)),axis=-1))+
                                          self.gamma2*tf.reduce_mean(tf.reduce_mean(tf.multiply(squared_distance(Z)/2.0,-squared_distance(self.stochastic_gates)),axis=1)),
                                 lambda: tf.constant(0.))
                
                loss = loss_fun  +  reg_gates + reg_sim
                
               
            else:
                loss = loss_fun
                reg_gates = tf.constant(0.)
                reg_sim = tf.constant(0.)
            # Get Optimizer
            if batch_normalization:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
            else:
                train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
            # For evaluation
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # Initialize the variables (i.e. assign their default value)
            init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        # Save into class members
        self.X = X
        self.y = y
        self.Z = Z
        #self.K_batch = K_batch
        self.pred = pred
        self.train_gates = train_gates
        self.is_train = is_train
        self.loss = loss
        self.reg_gates = reg_gates
        self.reg_sim = reg_sim
        self.pred_log = pred_log
        self.train_step = train_step
        self.correct_prediction = correct_prediction
        self.accuracy = accuracy
        self.output_node=output_node
        self.nnweights = nnweights.copy()
        self.weights=weights
        self.biases=biases
        # set random state
        tf.set_random_seed(seed)
        self.sess.run(init_op)
    
    #@property
    #def batch_size(self):
        #return self._batch_size
    
    #@batch_size.setter
    #def batch_size(self,value):
        #self._batch_size=value
        
    @property
    def activation_gating(self):
        return self._activation_gating
    
    @activation_gating.setter
    def activation_gating(self,value):
        if value == 'relu':
            self._activation_gating = tf.nn.relu                     
        elif value == 'l_relu':
            self._activation_gating = tf.nn.leaky_relu
        elif value == 'sigmoid':
            self._activation_gating = tf.nn.sigmoid
        elif value == 'tanh':
            self._activation_gating = tf.nn.tanh
        elif value == 'none':
            self._activation_gating = lambda x: x
        else:
            raise NotImplementedError('activation for the gating network not recognized')
    
    @property
    def activation_pred(self):
        return self._activation_pred
    
    @activation_pred.setter
    def activation_pred(self,value):
        if value == 'relu':
            self._activation_pred = tf.nn.relu                     
        elif value == 'l_relu':
            self._activation_pred = tf.nn.leaky_relu
        elif value == 'sigmoid':
            self._activation_pred = tf.nn.sigmoid
        elif value == 'tanh':
            self._activation_pred = tf.nn.tanh
        elif value == 'none':
            self._activation_pred = lambda x: x
        else:
            raise NotImplementedError('activation for the prediction network not recognized')
    
    
    def _to_tensor(self, x, dtype):
        """Convert the input `x` to a tensor of type `dtype`.
        # Arguments
            x: An object to be converted (numpy array, list, tensors).
            dtype: The destination type.
        # Returns
            A tensor.
        """
        return tf.convert_to_tensor(x, dtype=dtype)
    def get_weights(self):
        """
        Get network weights
        """
        weights_out=self.sess.run(self.nnweights,feed_dict={self.is_train:False})
        biases_out=self.sess.run(self.biases,feed_dict={self.is_train:False})
        return weights_out
    def hard_sigmoid(self, x, a):
        """Segment-wise linear approximation of sigmoid.
        Faster than sigmoid.
        Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
        In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.
        # Arguments
            x: A tensor or variable.
        # Returns
            A tensor.
        """
        x = a * x + 0.5
        zero = self._to_tensor(0., x.dtype.base_dtype)
        one = self._to_tensor(1., x.dtype.base_dtype)
        x = tf.clip_by_value(x, zero, one)
        return x

    def get_stochastic_gate_train(self, prev_x, train_gates):
        """
        This function replaced the feature_selector function in order to save Z
        """
        # gaussian reparametrization
        base_noise = tf.random_normal(shape=tf.shape(prev_x), mean=0., stddev=1.)
        
        z = self.alpha + self.sigma * base_noise * train_gates
        stochastic_gate = self.hard_sigmoid(z, self.a)
        
        return stochastic_gate
        

    def eval(self, new_X, new_y, new_Z,compute_sim):
        """
        Evaluate the accuracy and loss
        """
        
        acc, loss = self.sess.run([self.accuracy, self.loss], feed_dict={self.X: new_X,
                                                        self.y: new_y,
                                                        self.Z: new_Z,
                                                        self.train_gates: [0.0],
                                                        self.is_train:False,
                                                        self.compute_sim:compute_sim, 
                                                        })
        return np.squeeze(acc), np.squeeze(loss)

    def get_raw_alpha(self,X_in):
        """
        evaluate the learned parameter for stochastic gates 
        """
        dp_alpha = self.sess.run(self.alpha,feed_dict={self.X: X_in,self.is_train:False,})
        return dp_alpha

    def get_prob_alpha(self,X_in):
        """
        convert the raw alpha into the actual probability
        """
        dp_alpha = self.get_raw_alpha(X_in)
        prob_gate = self.compute_learned_prob(dp_alpha)
        return prob_gate

    def hard_sigmoid_np(self, x, a):
        return np.minimum(1, np.maximum(0,a*x+0.5))

    def compute_learned_prob(self, alpha):
        z = alpha
        stochastic_gate = self.hard_sigmoid_np(z, self.a)
        return stochastic_gate

    def load(self, model_path=None):
        if model_path == None:
            raise Exception()
        self.saver.restore(self.sess, model_path)

    def save(self, step, model_dir=None):
        if model_dir == None:
            raise Exception()
        try:
            os.mkdir(model_dir)
        except:
            pass
        model_file = model_dir + "/model"
        self.saver.save(self.sess, model_file, global_step=step)

    def train(self, 
              dataset, 
              batch_size,
              num_epoch=100,
              lr = 0.1,
              compute_sim=False,
              ):
                
        train_losses, train_accuracies = [], []
        val_losses = []
        val_accuracies = []
        
        # record the sim reg loss per batch
        sim_reg_losses_per_b = []
        # record the network weights
        nnweights_list = []
        print("num_samples : {}".format(dataset.num_samples))
        for epoch in range(num_epoch):
            avg_loss = 0.
            total_batch = int(dataset.num_samples/batch_size)
            
            
            # Deprecated: the compute_sim flag is up for every compute_sim_step
            # compute_sim = not ((epoch+1) % compute_sim_step)
            
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys,batch_zs = dataset.next_batch(batch_size)
                
                # for the 2nd penalty, precompute the K matrix 
                # TODO: make this squared_distance function consistent
                #(TODO: only compute it when feature_selection flag is on)
                #K_batch = 1.0 - squared_distance(self._to_tensor(batch_zs,dtype=tf.float32))/2.0
                #K_batch = 1.0 - distance_np(batch_zs)**2/2.0
                
                
                _,c = self.sess.run([self.train_step, self.loss],feed_dict={self.X: batch_xs,
                                                              self.y: batch_ys,
                                                              self.Z: batch_zs,
                                                              self.learning_rate : lr,
                                                              self.compute_sim: compute_sim,
                                                              self.train_gates: [1.0],
                                                              self.is_train:True,                                        
                                                              })
                
            
                #sim_reg_losses_per_b.append(reg_sim_fs)
                avg_loss += c / total_batch
            train_losses.append(avg_loss)
            # Display logs per epoch step
            if (epoch+1) % self.display_step == 0:
                
                valid_acc, valid_loss = self.eval(dataset.valid_data, dataset.valid_labels, dataset.valid_meta,compute_sim)
                val_accuracies.append(valid_acc)
                val_losses.append(valid_loss)
                #nnweights_list.append(nnweights_perE)       
                if self.output_node!=1:
                    print("Epoch: {} train loss={:.9f} valid loss= {:.9f} valid acc= {:.9f}".format(epoch+1,\
                                                                                                    avg_loss, valid_loss, valid_acc))
                else:
                    print("Epoch: {} train loss={:.9f} valid loss= {:.9f}".format(epoch+1,\
                                                                                  avg_loss, valid_loss))
                #print("train reg_fs: {}".format(reg_fs))                
                #print("train sim_penalty: {}".format(reg_sim_fs))
        print("Optimization Finished!")
        test_acc, test_loss = self.eval(dataset.test_data, dataset.test_labels, dataset.test_meta,compute_sim)
        print("test loss: {}, test acc: {}".format(test_loss, test_acc))
        self.acc=test_acc # used for recording test acc for figures
        #self.sim_reg_losses_per_b = sim_reg_losses_per_b # used for recording training sim loss
        self.train_losses2plot =  train_losses
        #self.nnweights_list = nnweights_list
        
        return train_accuracies, train_losses, val_accuracies, val_losses
                                       
    def test(self,X_test):
        """
        Predict on the test set
        """
        prediction = self.sess.run([self.pred], feed_dict={self.X: X_test,self.train_gates: [0.0],self.is_train:False,})
        if self.output_node!=1:
            prediction=np.argmax(prediction[0],axis=1)
        return prediction

    def evaluate(self, X, y, Z,compute_sim):
        """
        Get the test acc and loss
        """
        acc, loss = self.eval(X, y, Z,compute_sim)
        print("test loss: {}, test acc: {}".format(loss, acc))
        print("Saving model..")
        return acc, loss

    def get_KD(self,X,Z):     
        
        K_batch_sim = 1.0 - squared_distance(self._to_tensor(Z,dtype=tf.float32))/2.0       
        
        D_batch_sim = squared_distance(self._to_tensor(self.get_prob_alpha(X),dtype=tf.float32))
        
        K_batch_dis = squared_distance(self._to_tensor(Z,dtype=tf.float32))/2.0
        
        D_batch_dis = -squared_distance(self._to_tensor(self.get_prob_alpha(X),dtype=tf.float32))
        
        return K_batch_sim,D_batch_sim,K_batch_dis,D_batch_dis
