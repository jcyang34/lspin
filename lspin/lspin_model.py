import numpy as np
import tensorflow as tf
import os

class Model(object):
    def __init__(self, 
                 input_node,
                 hidden_layers_node,
                 output_node,
                 gating_net_hidden_layers_node,
                 learning_rate,
                 batch_size,
                 display_step, 
                 activation,
                 feature_selection=True,
                 batch_normalization=True,
                 a = 1,
                 sigma = 0.5,
                 lam = 0.5, 
                 stddev_input=0.1,
                 seed=1,
        ): 
        """ LSPIN Model
        # Arguments:
            input_node: integer, input dimension of the prediction network
            hidden_layers_node: list, number of nodes for each hidden layer for the prediction net, example: [200,200]
            output_node: integer, number of nodes for the output layer of the prediction net, 1 (regression) or 2 (classification)
            gating_net_hidden_layers_node: list, number of nodes for each hidden layer of the gating net, example: [200,200]
            learning_rate: float, learning rate of SGD
            batch_size: integer, batch size
            display_step: integer, number of epochs to output info
            activation: string, activation function of the prediction net: 'relu', 'l_relu', 'sigmoid', 'tanh', or 'none'
            feature_selection: bool, if using the gating net
            a: float, 
            sigma: float, std of the gaussion reparameterization 
            lam: float, regularization parameter of the L0 penalty
            stddev_input: float, std of the normal initializer for the network weights
            seed: integer, random seed
        """

        # Register hyperparameters of LSPIN
        self.a = a
        self.sigma = sigma
        self.lam = lam
        # Register hyperparameters for training
        self.lr = learning_rate
        self.batch_size = batch_size
        self.display_step = display_step

        G = tf.Graph()
        with G.as_default():
            self.sess = tf.Session(graph=G)
            # tf Graph Input
            X = tf.placeholder(tf.float32, [None, input_node]) # X.shape == [batch_size, feature_size]
            y = tf.placeholder(tf.float32, [None, output_node])
            train_gates = tf.placeholder(tf.float32,[1], name='train_gates')
            
            is_train = tf.placeholder(tf.bool,[], name='is_train') # for batch normalization
            
            self.gatesweights=[]
            self.nnweights = []
            prev_node = input_node
            prev_x = X
            
            # Gating network
            if feature_selection:
                for i in range(len(gating_net_hidden_layers_node)):
                    gates_layer_name = 'gate_layer' + str(i+1)
                    
                    with tf.variable_scope(gates_layer_name, reuse=tf.AUTO_REUSE):
                        weights = tf.get_variable('weights', [prev_node, gating_net_hidden_layers_node[i]],
                                                  initializer=tf.truncated_normal_initializer(stddev=stddev_input))
                        biases = tf.get_variable('biases', [gating_net_hidden_layers_node[i]],
                                                 initializer=tf.constant_initializer(0.0))
                    
                        self.gatesweights.append(weights)
                        self.gatesweights.append(biases)
                        
                        gates_layer_out = tf.nn.tanh(tf.matmul(prev_x,weights)+biases)

                        prev_node = gating_net_hidden_layers_node[i]
                        prev_x = gates_layer_out        
                weights2 = tf.get_variable('weights2', [prev_node,input_node],
                                                  initializer=tf.truncated_normal_initializer(stddev=stddev_input))
                biases2 = tf.get_variable('biases2', [input_node],
                                                 initializer=tf.constant_initializer(0.0))
  
                self.gatesweights.append(weights2)
                self.gatesweights.append(biases2)
                self.alpha= tf.nn.tanh(tf.matmul(prev_x,weights2)+biases2)
                prev_x = X
                prev_x = self.feature_selector(prev_x, train_gates)
                prev_node = input_node

            # Prediction network
            layer_name = 'layer' + str(1)
            for i in range(len(hidden_layers_node)):
                layer_name = 'layer' + str(i+1)
                with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                    weights = tf.get_variable('weights', [prev_node, hidden_layers_node[i]],
                                              initializer=tf.truncated_normal_initializer(stddev=stddev_input))
                    self.nnweights.append(weights)
                    biases = tf.get_variable('biases', [hidden_layers_node[i]],
                                             initializer=tf.constant_initializer(0.0))
                    self.nnweights.append(biases)
                    if batch_normalization:
                        prev_x = tf.layers.batch_normalization(prev_x, training=is_train)
                    layer_out = (tf.matmul(prev_x, weights) + biases)
               
                    if activation == 'relu':
                        layer_out = tf.nn.relu(layer_out)                        
                    elif activation == 'l_relu':
                        layer_out = tf.nn.leaky_relu(layer_out)
                    elif activation == 'sigmoid':
                        layer_out = tf.nn.sigmoid(layer_out)
                    elif activation == 'tanh':
                        layer_out = tf.nn.tanh(layer_out)
                    elif activation == 'none':
                        layer_out =(layer_out)
                    else:
                        raise NotImplementedError('activation not recognized')

                    prev_node = hidden_layers_node[i]
                    prev_x = layer_out

            # Output of model
            # Minimize error using cross entropy
            if output_node==1:
                # jcyang: n-1 layer can be any node now instead of just 1
                weights = tf.get_variable('weights', [prev_node, 1],
                                              initializer=tf.truncated_normal_initializer(stddev=stddev_input))
                self.nnweights.append(weights)
                biases = tf.get_variable('biases', [1],
                                         initializer=tf.constant_initializer(0.0))
                self.nnweights.append(biases)
                if batch_normalization:
                    layer_out = tf.layers.batch_normalization(layer_out, training=is_train)
                pred = (tf.matmul(layer_out, weights) + biases)
                loss_fun = tf.reduce_mean(tf.squared_difference(pred, y))
                pred_log = (layer_out)
            else:
                # jcyang: add the output layer here
                weights = tf.get_variable('weights', [prev_node, output_node],
                                              initializer=tf.truncated_normal_initializer(stddev=stddev_input))
                self.nnweights.append(weights)
                biases = tf.get_variable('biases', [output_node],
                                             initializer=tf.constant_initializer(0.0))
                self.nnweights.append(biases)
                
                if batch_normalization:
                    prev_x = tf.layers.batch_normalization(prev_x, training=is_train)
                layer_out = (tf.matmul(prev_x, weights) + biases)
                
                if activation == 'relu':
                    layer_out = tf.nn.relu(layer_out)                        
                elif activation == 'l_relu':
                    layer_out = tf.nn.leaky_relu(layer_out)
                elif activation == 'sigmoid':
                    layer_out = tf.nn.sigmoid(layer_out)
                elif activation == 'tanh':
                    layer_out = tf.nn.tanh(layer_out)
                elif activation == 'none':
                    layer_out =(layer_out)
                else:
                    raise NotImplementedError('activation not recognized')
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
                loss = loss_fun  +  reg_gates
                self.reg_gates = reg_gates
            else:
                loss = loss_fun
                self.reg_gates = tf.constant(0.)
            
            # Get Optimizer
            if batch_normalization:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
            else:
                train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
            # For evaluation
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # Initialize the variables (i.e. assign their default value)
            init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        # Save into class members
        self.X = X
        self.y = y
        self.pred = pred
        self.train_gates = train_gates
        self.is_train = is_train
        self.loss = loss
        self.pred_log = pred_log
        self.train_step = train_step
        self.correct_prediction = correct_prediction
        self.accuracy = accuracy
        self.output_node=output_node
        self.weights=weights
        self.biases=biases
        # set random state
        tf.set_random_seed(seed)
        self.sess.run(init_op)

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

    def feature_selector(self, prev_x, train_gates):
        '''
        feature selector - used at training time (gradients can be propagated)
        :param prev_x - input. shape==[batch_size, feature_num]
        :param train_gates (bool) - 1 during training, 0 during evaluation
        :return: gated input
        '''
        # gaussian reparametrization
        base_noise = tf.random_normal(shape=tf.shape(prev_x), mean=0., stddev=1.)
        
        z = self.alpha + self.sigma * base_noise * train_gates
        stochastic_gate = self.hard_sigmoid(z, self.a)
        
        new_x = prev_x * stochastic_gate
        return new_x

    def eval(self, new_X, new_y):
        """
        Evaluate the accuracy and loss
        """
        acc, loss = self.sess.run([self.accuracy, self.loss], feed_dict={self.X: new_X,
                                                        self.y: new_y,
                                                        self.train_gates: [0.0],
                                                        self.is_train:False,
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

    def train(self, trial, dataset, output_dir, num_epoch=100, plot_loss=False):
        train_losses, train_accuracies = [], []
        val_losses = []
        val_accuracies = []
        print("num_samples : {}".format(dataset.num_samples))
        for epoch in range(num_epoch):
            avg_loss = 0.
            total_batch = int(dataset.num_samples/self.batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = dataset.next_batch(self.batch_size)
                _, c, reg_fs = self.sess.run([self.train_step, self.loss, self.reg_gates], feed_dict={self.X: batch_xs,
                                                              self.y: batch_ys,
                                                              self.train_gates: [1.0],
                                                              self.is_train:True,                                        
                                                              })
                avg_loss += c / total_batch
            train_losses.append(avg_loss)
            # Display logs per epoch step
            if (epoch+1) % self.display_step == 0:
                valid_acc, valid_loss = self.eval(dataset.valid_data, dataset.valid_labels)
                val_accuracies.append(valid_acc)
                val_losses.append(valid_loss)
                                
                if self.output_node!=1:
                    print("Epoch: {} train loss={:.9f} valid loss= {:.9f} valid acc= {:.9f}".format(epoch+1,\
                                                                                                    avg_loss, valid_loss, valid_acc))
                else:
                    print("Epoch: {} train loss={:.9f} valid loss= {:.9f}".format(epoch+1,\
                                                                                  avg_loss, valid_loss))
                print("train reg_fs: {}".format(reg_fs))                
                
        #print("Optimization Finished!")
        test_acc, test_loss = self.eval(dataset.test_data, dataset.test_labels)
        #print("test loss: {}, test acc: {}".format(test_loss, test_acc))
        self.acc=test_acc # used for recording test acc for figures
        return train_accuracies, train_losses, val_accuracies, val_losses 
                                       
    def test(self,X_test):
        """
        Predict on the test set
        """
        prediction = self.sess.run([self.pred], feed_dict={self.X: X_test,self.train_gates: [0.0],self.is_train:False,})
        if self.output_node!=1:
            prediction=np.argmax(prediction[0],axis=1)
        return prediction

    def evaluate(self, X, y):
        """
        Get the test acc and loss
        """
        acc, loss = self.eval(X, y)
        #print("test loss: {}, test acc: {}".format(loss, acc))
        #print("Saving model..")
        return acc, loss

