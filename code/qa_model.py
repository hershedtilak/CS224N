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

logging.basicConfig(level=logging.INFO)

##### data should contain a list of sentences
def pad_sequences(data, max_length):
    ret_sen = []
    ret_length = []

    # Use this zero vector when padding sequences.
    zero_vector = PAD_ID

    for sentence in data:
        ### YOUR CODE HERE (~4-6 lines)
        truncatedLength = min(len(sentence),max_length)
        padding_size = max_length - truncatedLength
        newSentence = sentence[0:truncatedLength] + [zero_vector]*padding_size
        ret_sen.append(newSentence)
        ret_length.append(truncatedLength)
        ### END YOUR CODE ###
    return (ret_sen, ret_length)

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class GRUAttnCell(tf.nn.rnn_cell.GRUCell):
    def __init__(self, num_units, encoder_output, scope=None):
        self.hs = encoder_output
        super(GRUAttnCell, self).__init__(num_units)
    def __call__(self, inputs, state, scope=None):
        gru_out, gru_state = super(GRUAttnCell, self).__call__(inputs, state, scope)
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn"):
                ht = tf.nn.rnn_cell._linear(gru_out, self._num_units, True, 1.0)
                ht = tf.expand_dims(ht, axis=1)
            scores = tf.reduce_sum(self.hs*ht, reduction_indices=2, keep_dims=True)
            context = tf.reduce_sum(self.hs*scores, reduction_indices=1)
            with vs.variable_scope("AttnConcat"):
                out = tf.nn.relu(tf.nn.rnn_cell._linear([context, gru_out], self._num_units, True, 1.0))
            return (out, out)

class Encoder(object):
    def __init__(self, size, vocab_dim, config):
        self.size = size
        self.vocab_dim = vocab_dim
        self.config = config

    def encode(self, inputs, sequence_length, encoder_state_input, scope="encode"):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """    
        cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.flag.state_size, state_is_tuple=True)
        cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.flag.state_size, state_is_tuple=True)
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=sequence_length, initial_state_fw=encoder_state_input[0], initial_state_bw=encoder_state_input[1], dtype=tf.float32, parallel_iterations=None, swap_memory=True, time_major=False, scope=scope)
        concatOutputs = tf.concat(2, [outputs[0], outputs[1]])
        return concatOutputs, output_states

    def encode_w_attn(self, inputs, prev_states, scope="encode", reuse=False):
        self.attn_cell = GRUAttnCell(2*self.config.flag.state_size, prev_states)
        with vs.variable_scope(scope, reuse):
            outputs, output_states =  tf.nn.dynamic_rnn(self.attn_cell, inputs,  dtype=tf.float32)
        return outputs, output_states

class Decoder(object):
    def __init__(self, output_size, config):
        self.output_size = output_size
        self.config = config
        self.start_cell = tf.nn.rnn_cell.LSTMCell(self.config.flag.state_size, state_is_tuple=True)
        self.end_cell = tf.nn.rnn_cell.LSTMCell(self.config.flag.state_size, state_is_tuple=True)

    def decode(self, knowledge_rep):
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
        h_q, H_q, h_p, H_p = knowledge_rep
        with vs.variable_scope("answer_start"):
            outputs_start, a_s = tf.nn.dynamic_rnn(self.start_cell, H_p, dtype=tf.float32)
            a_s = tf.nn.rnn_cell._linear(a_s, self.config.flag.output_size, True, 1.0)
        with vs.variable_scope("answer_end"):
            outputs_end, a_e = tf.nn.dynamic_rnn(self.end_cell, outputs_start, dtype=tf.float32)
            a_e = tf.nn.rnn_cell._linear(a_e, self.config.flag.output_size, True, 1.0)
        
        return (a_s, a_e)

# TODO
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
        # ==== set up training/updating procedure ====
        
        self.saver = tf.train.Saver()

        pass


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        
        H_q, h_q  = self.encoder.encode(self.embeddings_q, self.sequence_length_q_placeholder, (None, None), scope="question")
        h_q_concat = tf.concat(1, [h_q[0].h, h_q[1].h])

        H_p_noAttn, h_p_noAttn = self.encoder.encode(self.embeddings_p, self.sequence_length_p_placeholder, h_q, scope="paragraph")
        
        H_p_attn, h_p_attn = self.encoder.encode_w_attn(H_p_noAttn, h_q_concat)
        
        knowledge_rep = (h_q_concat, H_q, h_p_attn, H_p_attn)
        self.a_s, self.a_e = self.decoder.decode(knowledge_rep)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        ##### LOSS ASSUMING OUTPUT IS PAIR OF TWO INTEGERS #####
        with vs.variable_scope("loss"):
            l1 = ssce(self.a_s, self.labels_answer_start)
            l2 = ssce(self.a_e, self.labels_answer_end)
            self.loss = tf.reduce_mean(l1+l2)

    def add_training_op(self, loss):
        with vs.variable_scope("loss"):
            optimizer = tf.train.AdamOptimizer(self.config.flag.learning_rate)
            #self.train_op = optimizer.minimize(loss)
            
            tuples = optimizer.compute_gradients(loss)
            grads = [entry[0] for entry in tuples]
            vars = [entry[1] for entry in tuples]
            grads, _ = tf.clip_by_global_norm(grads, self.config.flag.max_gradient_norm)
            clipped_gradients = zip(grads, vars)
            self.train_op = optimizer.apply_gradients(clipped_gradients)
            
    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        ##### Load embeddings - CURRENTLY USING LENGTH 50
        pretrained_embeddings = np.load(self.config.flag.data_dir + "/glove.trimmed.100.npz")
        # Do some stuff        
        with vs.variable_scope("embeddings"):
            embedding = tf.Variable(pretrained_embeddings['glove'], dtype=tf.float32)
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
        ## ASSUMING train_x is a tuple of (question, paragraph)
        input_feed[self.inputs_p_placeholder], input_feed[self.sequence_length_p_placeholder] = pad_sequences(train_x[0], self.config.flag.max_size_p)
        input_feed[self.inputs_q_placeholder], input_feed[self.sequence_length_q_placeholder] = pad_sequences(train_x[1], self.config.flag.max_size_q)
        
        input_feed[self.labels_answer_start] = [item[0] for item in train_y]
        input_feed[self.labels_answer_end] = [item[1] for item in train_y]

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        output_feed = [self.train_op, self.loss]

        outputs = session.run(output_feed, feed_dict=input_feed)
        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        input_feed[self.inputs_p_placeholder], input_feed[self.sequence_length_p_placeholder] = pad_sequences(valid_x[0], self.config.flag.max_size_p)
        input_feed[self.inputs_q_placeholder], input_feed[self.sequence_length_q_placeholder] = pad_sequences(valid_x[1], self.config.flag.max_size_q)
        
        input_feed[self.labels_answer_start] = [item[0] for item in valid_y]
        input_feed[self.labels_answer_end] = [item[1] for item in valid_y]
        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x
        ## Here, output feed should represent want we want to get from the session, in this case it should
        ## what the system predicts
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
        input_feed[self.inputs_p_placeholder], input_feed[self.sequence_length_p_placeholder] = pad_sequences(test_x[0], self.config.flag.max_size_p)
        input_feed[self.inputs_q_placeholder], input_feed[self.sequence_length_q_placeholder] = pad_sequences(test_x[1], self.config.flag.max_size_q)

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

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
        #average over num examples
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

        for i in range(sample):
            start, end = self.answer(session, ([dataset[datatype][0][i]], [dataset[datatype][1][i]], [dataset[datatype][2][i]]) )
            prediction = ' '.join(ground_truth[0][i][start[0]:end[0]])
            gt = ' '.join(ground_truth[1][i])
            f1_instance = f1_score(prediction, gt)
            em_instance = exact_match_score(prediction, gt)
            em = em + em_instance
            f1 = f1 + f1_instance
        em = 100 * em / float(sample)
        f1 = 100 * f1 / float(sample)
        
        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))
        
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

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training
       
        # TODO - Figure this out        
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
        num_train = len(dataset['train'][2])
        num_train = 50
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
                _, batch_loss = self.optimize(session, (batchP, batchQ), batchA)
                loss += batch_loss
                count += 1
                #if count % 1000:
                #    logging.info("Batch Loss: %f\n", batch_loss)
                
            logging.info("Loss for epoch " + str(k+1) + ": " + str(float(loss) / count))
            self.evaluate_answer(session, dataset, self.config.flag.evaluate, log=True)
            save_path = self.saver.save(session, train_dir + "/" + str(int(tic)) + "_epoch" + str(k) + ".ckpt")
            #print(save_path)
            


