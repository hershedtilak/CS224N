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
from qa_config import Config

#my imports
from tensorflow.python.ops.nn import sparse_softmax_cross_entropy_with_logits as ssce
from qa_data import PAD_ID

logging.basicConfig(level=logging.INFO)

##### data should contain a list of sentences
def pad_sequences(data, max_length):
    ret = []

    # Use this zero vector when padding sequences.
    zero_vector = [PAD_ID] * Config.n_features

    for sentence in data:
        ### YOUR CODE HERE (~4-6 lines)
        truncatedLength = min(len(sentence),max_length)
        padding_size = max_length - truncatedLength
        newSentence = sentence[0:truncatedLength] + [zero_vector]*padding_size
        maskingSeq = [True]*truncatedLength + [False]*padding_size
        ret.append((newSentence, maskingSeq))
        ### END YOUR CODE ###
    return ret

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, size, vocab_dim, config):
        self.size = size
        self.vocab_dim = vocab_dim
        self.config = config

    def encode(self, inputs, masks, encoder_state_input):
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
        cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.lstm_size_en)
        cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.lstm_size_en)
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, tf.boolean_mask(self.inputs, masks), sequence_length=None, initial_state_fw=None, initial_state_bw=None, dtype=None, parallel_iterations=None, swap_memory=False, time_major=False, scope="encode")
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
        

        return

# TODO
class QASystem(object):
    def __init__(self, encoder, decoder, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        # ==== set up placeholder tokens ========
        self.inputs_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.max_length, 1), name="x")
        self.labels_placeholder = tf.placeholder(tf.float32, shape=(None, 1), name="y")

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        pass


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        raise NotImplementedError("Connect all parts of your system here!")


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        ##### LOSS ASSUMING OUTPUT IS PAIR OF TWO INTEGERS #####
        with vs.variable_scope("loss"):
            l1 = ssce(self.a_s, self.start_answer)
            l2 = ssce(self.a_e, self.end_answer)
            self.loss = l1 + l2

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
            lookup_q = tf.nn.embedding_lookup(embedding, self.inputs_q)
            lookup_p = tf.nn.embedding_lookup(embedding, self.inputs_p)
            self.embeddings_q = tf.reshape(lookup_q, [-1, self.config.flag.max_length, self.config.flag.embedding_size])
            self.embeddings_p = tf.reshape(lookup_p, [-1, self.config.flag.max_length, self.config.flag.embedding_size])
        
    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = []

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

        output_feed = []

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
