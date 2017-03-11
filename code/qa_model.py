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
# from tensorflow.python.ops.nn import sparse_softmax_cross_entropy_with_logits as ssce
from qa_data import PAD_ID

logging.basicConfig(level=logging.INFO)

##### data should contain a list of sentences
def pad_sequences(data, max_length):
    ret_sen = []
    ret_length = []

    # Use this zero vector when padding sequences.
    zero_vector = [PAD_ID] * Config.n_features

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

    def encode(self, inputs, sequence_length, encoder_state_input):
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
        cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.flag.state_size)
        cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.flag.state_size)

        print(inputs.get_shape())

        (fw_out, bw_out), output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=sequence_length, initial_state_fw=encoder_state_input, initial_state_bw=encoder_state_input, dtype=tf.float32, parallel_iterations=None, swap_memory=False, time_major=True, scope="encode")
        
        return outputs, output_states

    def encode_w_attn(self, inputs, prev_states, scope="encode", reuse=False):
        self.attn_cell = GRUAttnCell(self.config.flag.state_size, prev_states)
        with vs.variable_scope(scope, reuse):
            outputs, output_states =  dynamic_rnn(self.attn_cell,inputs)
        return outputs, output_states

class Decoder(object):
    def __init__(self, output_size, config):
        self.output_size = output_size
        self.config = config

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
        with vs.scope("answr_start"):
            a_s = tf.nn.rnn_cell._linear([h_q, h_p], output_size=self.config.flag.output_size)
        with vs.scope("answer_end"):
            a_e = tf.nn.rnn_cell._linear([h_q, h_p], output_size=self.config.flag.output_size)
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
        self.inputs_p_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.flag.max_size_p, config.flag.embedding_size), name="inputs_p_placeholder")
        self.inputs_q_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.flag.max_size_q, config.flag.embedding_size), name="inputs_q_placeholder")
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
        pass


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        # raise NotImplementedError("Connect all parts of your system here!")
        h_q, H_q = self.encoder.encode(self.embeddings_q, self.sequence_length_q_placeholder, tf.zeros(self.config.flag.state_size))
        h_p, H_p = self.encoder.encode_w_attn(self.embeddings_p, tf.zeros(self.config.flag.state_size))
        knowledge_rep = (h_q, H_q, h_p, H_p)
        self.a_s, self.a_e = self.decoder.decode(knowledge_rep)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        ##### LOSS ASSUMING OUTPUT IS PAIR OF TWO INTEGERS #####
        with vs.variable_scope("loss"):
            l1 = tf.python.ops.nn.sparse_softmax_cross_entropy_with_logits(self.a_s, self.start_answer_placeholder)
            l2 = tf.python.ops.nn.sparse_softmax_cross_entropy_with_logits(self.a_e, self.end_answer_placeholder)
            self.loss = l1+l2

    def add_training_op(self, loss):
        with vs.variable_scope("loss"):
            optimizer = tf.train.AdamOptimizer(Config.lr)
            self.train_op = optimizer.minimize(loss)
            
    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        ##### Load embeddings - CURRENTLY USING LENGTH 100
        pretrained_embeddings = np.load(self.config.flag.data_dir + "/glove.trimmed.100.npz")
        # Do some stuff        
        with vs.variable_scope("embeddings"):
            embedding = tf.Variable(pretrained_embeddings['glove'])
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
        input_feed[self.inputs_p_placeholder], _ = pad_sequences(train_x[0])
        input_feed[self.inputs_q_placeholder], input_feed[self.sequence_length_q_placeholder] = pad_sequences(train_x[1])
        
        input_feed[self.start_answer_placeholder] = train_y[0]
        input_feed[self.end_answer_placeholder] = train_y[1]

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        output_feed = [self.train_op, self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        input_feed[self.input_p] = valid_x[0]
        input_feed[self.input_q] = valid_x[1]
        
        input_feed[self.output] = valid_y
        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x
        ## Here, output feed should represent want we want to get from the session, in this case it should
        ## what the system predicts
        output_feed = [self.a_s, self.a_e]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

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
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
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
        for p, q, a in dataset['train']:
            self.optimize(session, (p, q), a)

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
        num_train = len(dataset['train'][2])

        for i in range(self.config.flag.epochs):
            # TODO shuffle data
            for p, q, a in dataset['train']:
                loss = self.optimize(session, (p,q), a)
                # print loss
                break



