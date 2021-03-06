from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score

#my imports
from tensorflow.python.ops.nn import sparse_softmax_cross_entropy_with_logits as ssce
from qa_data import PAD_ID
import random
import os
from tqdm import *
from tensorflow.python.ops import array_ops

logging.basicConfig(level=logging.INFO)

def pad_sequences(data, max_length):
    ret_sen = []
    ret_length = []
    ret_mask = []

    # Use this zero vector when padding sequences.
    zero_vector = PAD_ID

    for sentence in data:
        truncatedLength = min(len(sentence),max_length)
        padding_size = max_length - truncatedLength
        newSentence = sentence[0:truncatedLength] + [zero_vector]*padding_size
        maskingSeq = [True]*truncatedLength + [False]*padding_size
        ret_sen.append(newSentence)
        ret_length.append(truncatedLength)
        ret_mask.append(maskingSeq)
    return (ret_sen, ret_length, ret_mask)

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

def _reverse(input_, seq_lengths, seq_dim, batch_dim):
    if seq_lengths is not None:
        return array_ops.reverse_sequence(
            input=input_, seq_lengths=seq_lengths,
            seq_dim=seq_dim, batch_dim=batch_dim)
    else:
        return array_ops.reverse(input_, axis=[seq_dim])
    

class MatchLSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, config, Hq, LSTMCell, scope=None, reuse=None):
        self.config = config
        self._state_size = self.config.flag.state_size
        self.max_q_size = self.config.flag.max_size_q
        self.batch_size = self.config.flag.batch_size
        self.Hq = Hq
        self.scope = scope
        self.reuse = reuse
        self.cell = LSTMCell
        
    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size
        
    def __call__(self, inputs, state, scope="match_lstm"):
        dims = tf.shape(inputs)
                
        with tf.variable_scope(scope, reuse=self.reuse):
            dh = self._state_size
            xinit = tf.contrib.layers.xavier_initializer()
            
            # All of these variables are actually the transpose of what is said in the paper except w
            W_q = tf.get_variable("W_q", [2*dh, dh], initializer=tf.contrib.layers.xavier_initializer(), dtype=np.float32)
            W_p = tf.get_variable("W_p", [dh, dh], initializer=tf.contrib.layers.xavier_initializer(), dtype=np.float32)
            W_r = tf.get_variable("W_r", [dh, dh], initializer=tf.contrib.layers.xavier_initializer(), dtype=np.float32)
            b_p = tf.get_variable("b_p", [1, dh], initializer=tf.uniform_unit_scaling_initializer(), dtype=np.float32)
            w = tf.get_variable("w", [dh,1], initializer=tf.uniform_unit_scaling_initializer(), dtype=np.float32)
            b = tf.get_variable("b", [1, 1], initializer=tf.uniform_unit_scaling_initializer(), dtype=np.float32)

            # right_side_G is (?,20,50) (will be broadcast)
            right_arg = tf.matmul(state[1], W_r) + tf.tile(b_p, [dims[0],1])
            left_arg = tf.matmul(inputs, W_p)
            right_side_G = left_arg + right_arg
            right_side_G = tf.reshape(right_side_G, [-1, 1, dh])
            right_side_G = tf.tile(right_side_G, [1, self.max_q_size, 1])
            
            # left_side_G is (?,20,50)
            Hq = tf.reshape(self.Hq, [-1, 2*dh])
            left_side_G = tf.reshape(tf.matmul(Hq, W_q), [-1, self.max_q_size, dh])
            
            # G (which is actually the transpose of G in the paper) is (?,20,50)
            G = tf.tanh(left_side_G + right_side_G)
            
            # alpha_i_reshaped is the correct alpha (not transposed) and is (?, 1, 20)
            G_reshaped = tf.reshape(G, [-1, dh])
            firstTerm = tf.reshape(tf.matmul(G_reshaped, w), [-1, 1, self.max_q_size])
            secondTerm = b

            alpha_i = tf.nn.softmax(firstTerm + secondTerm)
            
            # bottom terms
            Hq_alpha = tf.reshape(tf.batch_matmul(alpha_i, self.Hq), [-1, 2*dh])
            
            # z is (?, 100)
            z = tf.concat(1, [inputs, Hq_alpha])
        
        # pass z through appropriate LSTM
        with tf.variable_scope(self.scope, reuse=False):
            output, new_state = self.cell(z, state)
            
        # updates for next iteration
        return output, (new_state.c, new_state.h)
    

class Encoder(object):
    def __init__(self, size, vocab_dim, config):
        self.size = size
        self.vocab_dim = vocab_dim
        self.config = config
        
    def encodeQ(self, inputs, seq_len, dropout, scope="encodeQ"):
    
        q_cell_fwd = tf.nn.rnn_cell.LSTMCell(self.config.flag.state_size, state_is_tuple=True, initializer=tf.contrib.layers.xavier_initializer())
        q_cell_bwd = tf.nn.rnn_cell.LSTMCell(self.config.flag.state_size, state_is_tuple=True, initializer=tf.contrib.layers.xavier_initializer())
        q_cell_fwd_dropout = tf.nn.rnn_cell.DropoutWrapper(q_cell_fwd, 1.0, 1-dropout)
        q_cell_bwd_dropout = tf.nn.rnn_cell.DropoutWrapper(q_cell_bwd, 1.0, 1-dropout)
        with vs.variable_scope(scope):
            (fwd_out, bwd_out), outputStates = tf.nn.bidirectional_dynamic_rnn(q_cell_fwd_dropout, q_cell_bwd_dropout, inputs, sequence_length=seq_len, dtype=tf.float32)
        outputs = tf.concat(2, [fwd_out, bwd_out])
        
        return outputs, outputStates
        
    def encodeP(self, inputs, seq_len, dropout, scope="encodeP"):
        p_cell = tf.nn.rnn_cell.LSTMCell(self.config.flag.state_size, state_is_tuple=True, initializer=tf.contrib.layers.xavier_initializer())
        p_cell_dropout = tf.nn.rnn_cell.DropoutWrapper(p_cell, 1.0, 1-dropout)
        with vs.variable_scope(scope):
            outputs, outputStates = tf.nn.dynamic_rnn(p_cell_dropout, inputs, sequence_length=seq_len, dtype=tf.float32)
        return outputs, outputStates
    
    def encodeMatchLSTM(self, inputs, Hq, seq_len, dropout, scope="encodeMatchLSTM"):
        dh = self.config.flag.state_size
        dims = tf.shape(inputs)
        
        fwdLSTMCell = tf.nn.rnn_cell.LSTMCell(dh, state_is_tuple=True, initializer=tf.contrib.layers.xavier_initializer())
        bwdLSTMCell = tf.nn.rnn_cell.LSTMCell(dh, state_is_tuple=True, initializer=tf.contrib.layers.xavier_initializer())  

        fwdLSTMCell_dropout = tf.nn.rnn_cell.DropoutWrapper(fwdLSTMCell, 1.0, 1-dropout)
        bwdLSTMCell_dropout = tf.nn.rnn_cell.DropoutWrapper(bwdLSTMCell, 1.0, 1-dropout)
    
        matchLSTMCellFwd = MatchLSTMCell(self.config, Hq, fwdLSTMCell_dropout, "fwd", False)
        matchLSTMCellBwd = MatchLSTMCell(self.config, Hq, bwdLSTMCell_dropout, "bwd", True)
        
        initState = tf.tile(tf.constant(0, dtype=tf.float32, shape=[1, dh]), [dims[0], 1])
        initStateTuple = (initState, initState)
        
        inputsRev = _reverse(inputs, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
        
        with vs.variable_scope(scope):
            outputsFwd, outputStatesFwd = tf.nn.dynamic_rnn(matchLSTMCellFwd, inputs, initial_state=initStateTuple, sequence_length=seq_len, dtype=tf.float32)
            outputsBwd, outputStatesBwd = tf.nn.dynamic_rnn(matchLSTMCellBwd, inputsRev, initial_state=initStateTuple, sequence_length=seq_len, dtype=tf.float32)
        
        outputsBwd = _reverse(outputsBwd, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
        outputStatesBwd = _reverse(outputStatesBwd, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
        
        outputs = tf.concat(2, [outputsFwd, outputsBwd])
        outputStates = tf.concat(2, [outputStatesFwd, outputStatesBwd])
        
        # outputs is Hr transposed
        return outputs, outputStates
   
  
class Decoder(object):
    def __init__(self, output_size, config):
        self.output_size = output_size
        self.config = config
        self.cell = tf.nn.rnn_cell.LSTMCell(self.config.flag.state_size, state_is_tuple=True)

    def decodeAnswerPtr(self, Hr, seq_len, scope="ptr_network"):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.
        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        
        dh = self.config.flag.state_size
        batch_size = tf.shape(Hr)[0]
        p_size = self.config.flag.max_size_p

        initState = tf.tile(tf.constant(0, dtype=tf.float32, shape=[1, dh]), [batch_size, 1])
        h1 = (initState, initState)
        
        with tf.variable_scope(scope):
            xinit = tf.contrib.layers.xavier_initializer()
            
            # All of these variables are actually the transpose of what is said in the paper except v
            V = tf.get_variable("V", [2*dh, dh], initializer=tf.contrib.layers.xavier_initializer(), dtype=np.float32)
            W_a = tf.get_variable("W_a", [dh, dh], initializer=tf.contrib.layers.xavier_initializer(), dtype=np.float32)
            b_a = tf.get_variable("b_a", [1, dh], initializer=tf.uniform_unit_scaling_initializer(), dtype=np.float32)
            v = tf.get_variable("v", [dh,1], initializer=tf.uniform_unit_scaling_initializer(), dtype=np.float32)
            c = tf.get_variable("c", [1, 1], initializer=tf.uniform_unit_scaling_initializer(), dtype=np.float32)

            ############# start index #############
            # calculate F transpose (?,p_size,dh)
            Hr_reshaped = tf.reshape(Hr, [-1, 2*dh])
            left_arg = tf.matmul(Hr_reshaped, V)
            left_arg = tf.reshape(left_arg, [-1, p_size, dh])
            right_arg = tf.matmul(h1[1], W_a) + tf.tile(b_a, [batch_size,1])
            right_arg = tf.expand_dims(right_arg, axis=1)
            right_arg = tf.tile(right_arg, [1, p_size, 1])
            F1 = tf.tanh(left_arg + right_arg)
            F1_reshaped = tf.reshape(F1, [-1, dh])
            
            # calculate Beta 0
            firstTerm = tf.reshape(tf.matmul(F1_reshaped, v), [-1,p_size])
            secondTerm = c
            beta_0_out = firstTerm + secondTerm
            beta_0 = tf.nn.softmax(beta_0_out)
            
            z1 = tf.reshape(tf.batch_matmul(tf.expand_dims(beta_0, axis=1), Hr), [-1,2*dh])

            # pass z through appropriate LSTM
            output, new_state = self.cell(z1, h1)
            h2 = (new_state.c, new_state.h)
            
            ########### end index #################
            # calculate F transpose
            right_arg2 = tf.matmul(h2[1], W_a) + tf.tile(b_a, [batch_size,1])
            right_arg2 = tf.expand_dims(right_arg2, axis=1)
            right_arg2 = tf.tile(right_arg2, [1, p_size, 1])
            F2 = tf.tanh(left_arg + right_arg2)
            F2_reshaped = tf.reshape(F2, [-1, dh])
            
            # calculate Beta 0
            firstTerm2 = tf.reshape(tf.matmul(F2_reshaped, v), [-1,p_size])
            secondTerm2 = c
            beta_1_out = firstTerm2 + secondTerm2
            beta_1 = tf.nn.softmax(beta_1_out)
            
            z2 = tf.reshape(tf.batch_matmul(tf.expand_dims(beta_1, axis=1), Hr), [-1,2*dh])

            # pass z through appropriate LSTM
            tf.get_variable_scope().reuse_variables()
            output2, new_state2 = self.cell(z2, h2)
            
        # get a_s and a_e from beta matrix
        a_s = beta_0_out
        a_e = beta_1_out
        return (a_s, a_e)


class QASystem(object):
    def __init__(self, encoder, decoder, config=None, *args):
        """
        Initializes your System
        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        # ==== set up placeholder tokens ========
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.mask_placeholder = tf.placeholder(tf.bool, shape=(None, self.config.flag.max_size_p), name="mask")
        self.inputs_p_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.flag.max_size_p), name="inputs_p_placeholder")
        self.inputs_q_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.flag.max_size_q), name="inputs_q_placeholder")
        self.sequence_length_p_placeholder = tf.placeholder(tf.int32, shape=None, name="sequence_length_p")
        self.sequence_length_q_placeholder = tf.placeholder(tf.int32, shape=None, name="sequence_length_q")
        self.labels_answer_start = tf.placeholder(tf.int32, shape=None, name="answer_start")
        self.labels_answer_end = tf.placeholder(tf.int32, shape=None, name="answer_end")
        self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout")

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()
            self.add_training_op(self.loss)
        
        self.saver = tf.train.Saver()


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        
        # LSTM Preprocessing Layer
        Hq, _ = self.encoder.encodeQ(self.embeddings_q, self.sequence_length_q_placeholder, self.dropout_placeholder)
        Hp, _ = self.encoder.encodeP(self.embeddings_p, self.sequence_length_p_placeholder, self.dropout_placeholder)
                
        # Match LSTM layer
        Hr, _ = self.encoder.encodeMatchLSTM(Hp, Hq, self.sequence_length_p_placeholder, self.dropout_placeholder)
        
        # Answer Pointer Layer
        self.a_s, self.a_e = self.decoder.decodeAnswerPtr(Hr, self.sequence_length_p_placeholder)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        self.a_s = tf.add(self.a_s, (1 - tf.cast(self.mask_placeholder, 'float')) * (-1e30), name="exp_mask")
        self.a_e = tf.add(self.a_e, (1 - tf.cast(self.mask_placeholder, 'float')) * (-1e30), name="exp_mask")
        with vs.variable_scope("loss"):
            l1 = ssce(self.a_s, self.labels_answer_start)
            l2 = ssce(self.a_e, self.labels_answer_end)
            self.loss = tf.reduce_mean(l1+l2)

    def add_training_op(self, loss):
        with vs.variable_scope("loss"):
            g_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.config.flag.learning_rate, g_step,int(40000)/self.config.flag.batch_size, self.config.flag.step_decay_rate, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            tuples = optimizer.compute_gradients(loss)
            grads = [entry[0] for entry in tuples]
            vars = [entry[1] for entry in tuples]
            self.outGrad = tf.global_norm(grads)        # FOR DEBUGGING
            grads, _ = tf.clip_by_global_norm(grads, self.config.flag.max_gradient_norm)
            clipped_gradients = zip(grads, vars)
            self.learning_rate = learning_rate          # FOR DEBUGGING

            self.train_op = optimizer.apply_gradients(clipped_gradients, global_step=g_step)
            
    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        pretrained_embeddings = np.load(self.config.flag.data_dir + "/glove.trimmed.100.npz")
        with vs.variable_scope("embeddings"):
            #embedding = tf.Variable(pretrained_embeddings['glove'], dtype=tf.float32)
            embedding = tf.constant(pretrained_embeddings['glove'], dtype=tf.float32)
            lookup_q = tf.nn.embedding_lookup(embedding, self.inputs_q_placeholder)
            lookup_p = tf.nn.embedding_lookup(embedding, self.inputs_p_placeholder)
            self.embeddings_q = tf.reshape(lookup_q, [-1, self.config.flag.max_size_q, self.config.flag.embedding_size])
            self.embeddings_p = tf.reshape(lookup_p, [-1, self.config.flag.max_size_p, self.config.flag.embedding_size])
        
    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}
        input_feed[self.inputs_p_placeholder], input_feed[self.sequence_length_p_placeholder], input_feed[self.mask_placeholder] = pad_sequences(train_x[0], self.config.flag.max_size_p)
        input_feed[self.inputs_q_placeholder], input_feed[self.sequence_length_q_placeholder], _ = pad_sequences(train_x[1], self.config.flag.max_size_q)
        input_feed[self.labels_answer_start] = [item[0] for item in train_y]
        input_feed[self.labels_answer_end] = [item[1] for item in train_y]
        input_feed[self.dropout_placeholder] = self.config.flag.dropout

        output_feed = [self.train_op, self.loss, self.outGrad] #, self.a_s, self.a_e, self.labels_answer_start, self.labels_answer_end]

        outputs = session.run(output_feed, feed_dict=input_feed)
        
        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        input_feed[self.inputs_p_placeholder], input_feed[self.sequence_length_p_placeholder], input_feed[self.mask_placeholder] = pad_sequences(valid_x[0], self.config.flag.max_size_p)
        input_feed[self.inputs_q_placeholder], input_feed[self.sequence_length_q_placeholder], _ = pad_sequences(valid_x[1], self.config.flag.max_size_q)
        input_feed[self.labels_answer_start] = [item[0] for item in valid_y]
        input_feed[self.labels_answer_end] = [item[1] for item in valid_y]
        input_feed[self.dropout_placeholder] = 0
        
        output_feed = [self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}
        input_feed[self.inputs_p_placeholder], input_feed[self.sequence_length_p_placeholder], input_feed[self.mask_placeholder] = pad_sequences(test_x[0], self.config.flag.max_size_p)
        input_feed[self.inputs_q_placeholder], input_feed[self.sequence_length_q_placeholder], _ = pad_sequences(test_x[1], self.config.flag.max_size_q)
        input_feed[self.dropout_placeholder] = 0
        
        output_feed = [self.a_s, self.a_e]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.
        This method calls self.test() which explicitly calculates validation cost.
        How you implement this function is dependent on how you design
        your data iteration function
        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = valid_cost + self.test(sess, valid_x, valid_y)
        # average over num examples
        valid_cost = float(valid_cost)/len(valid_dataset)

        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False, datatype='val'):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels
        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.
        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.
        fname = "../.."
        with open(os.path.join(self.config.flag.data_dir, "%s.context"%datatype)) as f:
            data_paragraph = [line.split() for line in f.read().splitlines()]
        with open(os.path.join(self.config.flag.data_dir, "%s.answer"%datatype)) as f:
            data_answer = [line.split() for line in f.read().splitlines()]
        ground_truth= (data_paragraph, data_answer)

        i=0
        while i < sample:
            preds = self.answer(session, (dataset[datatype][0][i:i+self.config.flag.batch_size],dataset[datatype][1][i:i+self.config.flag.batch_size],dataset[datatype][2][i:i+self.config.flag.batch_size]))
            for j in range(len(preds[0])):
                prediction = ' '.join(ground_truth[0][i][preds[0][j]:(preds[1][j]+1)])
                gt = ' '.join(ground_truth[1][i])
                f1_instance = f1_score(prediction, gt)
                em_instance = exact_match_score(prediction, gt)
                em = em + em_instance
                f1 = f1 + f1_instance
                i += 1
        em = 100 * em / float(sample)
        f1 = 100 * f1 / float(sample)
        
        if log:
            logging.info("Output for '{}' dataset - F1: {}, EM: {}, for {} samples".format(datatype, f1, em, sample))
        
        return f1, em

    def train(self, session, dataset, train_dir):
        """
        Implement main training loop
        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.
        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one
        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.
        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.
        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """
      
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
        num_train = len(dataset['train'][2])
        #num_train = self.config.flag.evaluate
        batch_size = self.config.flag.batch_size
        batch = range(num_train)
        for k in range(self.config.flag.epochs):
            logging.info("\n===== EPOCH " + str(k+1) + " =====")
            # TODO shuffle data
            loss = 0 
            count = 0
            random.shuffle(batch)
            for i in tqdm(range(0,num_train,batch_size)):
                if(i+batch_size > len(batch)):
                    indices = batch[i:]
                else:
                    indices = batch[i:i+batch_size]
                batchP = [dataset['train'][0][j] for j in indices]
                batchQ = [dataset['train'][1][j] for j in indices]
                batchA = [dataset['train'][2][j] for j in indices]
                _, batch_loss, grad_norm = self.optimize(session, (batchP, batchQ), batchA)
                loss += batch_loss
                count += 1
                print("Batch Loss: {}, Gradient: {}".format(batch_loss, grad_norm))
                #print("Predicted a_s={} a_e={}".format(np.argmax(a_s, axis=1),np.argmax(a_e, axis=1)))
                #print("Actual a_s={} a_e={}".format(s,e))
                
            logging.info("Loss for epoch " + str(k+1) + ": " + str(float(loss) / count))
            self.evaluate_answer(session, dataset, self.config.flag.evaluate, log=True, datatype='train')
            self.evaluate_answer(session, dataset, self.config.flag.evaluate, log=True, datatype='val')
            save_path = self.saver.save(session, train_dir + "/" + str(int(tic)) + "_epoch" + str(k) + ".ckpt")
            #print(save_path)
