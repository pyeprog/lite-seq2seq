import tensorflow as tf
import os
from random import random

MODEL_PATH = './models'
if not os.path.isdir(MODEL_PATH):
    os.mkdir(MODEL_PATH)

EMBEDDING_DIM = 128
ENCODER_RNN_SIZE = 256
ENCODER_RNN_LAYERS_N = 2

class Seq2seq:
    def __init__(self):
        self._id = (str(random())[2:] + str(random())[2:])[:20]
        self.model_ckpt_dir = os.path.join(MODEL_PATH, self._id)
        self.model_ckpt_path = os.path.join(self.model_ckpt_dir, 'checkpoint.ckpt')

        if not os.path.isdir(self.model_ckpt_dir):
            os.mkdir(self.model_ckpt_dir)

    def _text_file_parse(self, path):
        '''
        Read text content from path, encode words into int
        @param path: str, the path of file
        @return:
            int_contents: list(list(int)), encoded int list
            int_to_vocab: dict(int:str), int to vocab dict
        '''

        self.special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
        with open(path, 'r') as fp:
            text_contents = fp.read()
        
        text_contents = text_contents.split('\n')
        vocab_list = self.special_words + [word for line in text_contents for word in line.split()]
        vocab_list = list(set(vocab_list))

        int_to_vocab = {i:word for i, word in enumerate(vocab_list)}
        vocab_to_int = {word:i for i, word in int_to_vocab.items()}

        unk_word_id = vocab_to_int['<UNK>']
        int_contents = [[vocab_to_int.get(word, unk_word_id) for word in line.split()] for line in text_contents]
        return int_contents, int_to_vocab


    def _rnn_cell(rnn_size):
        return tf.nn.rnn_cell.GRUCell(rnn_size
                initializer=tf.random_uniform_initializer(-0.1, 0.1))


    def train(self, encode_file_path, decode_file_path):
        pass


    def train_new(self, encode_file_path, decode_file_path):
        '''
        Train the model for the first time or retrain the model
        @encode_file_path: str, the path of the encoder training file
        @decode_file_path: str, the path of the decoder training file
        @return: None
        '''
        encode_seqs, encoder_int_to_vocab = self._text_file_parse(encode_file_path)
        decode_seqs, decoder_int_to_vocab = self._text_file_parse(decode_file_path)

        self.graph = tf.Graph()
        with self.graph.as_default():
            # create placeholder
            ## why the shape is [None, None]? explain
            encoder_input = tf.placeholder(tf.int32, shape=[None, None], name='input')
            decoder_target = tf.placeholder(tf.int32, shape=[None, None], name='targets')
            
            ## why does it need sequence length placeholder? explain
            encoder_input_seq_lengths = tf.placeholder(tf.int32, shape=[None,], name='source_lens')
            decoder_target_seq_lengths = tf.placeholder(tf.int32, shape=[None,], name='target_lens')

            # Build whole model and Get training_logits
            with tf.Variable_scope('encoder'):
                encoder_wordvec = tf.contrib.layers.embed_sequence(encoder_input, len(encoder_int_to_vocab), EMBEDDING_DIM)
                rnn_cell_list = [self._rnn_cell(ENCODER_RNN_SIZE) for _ in range(ENCODER_RNN_LAYERS_N)]
                encoder_rnn = tf.nn.rnn_cell.MultiRNNCell(rnn_cell_list)
                encoder_output, encoder_final_state = tf.nn.dynamic_rnn(encoder_rnn, encoder_wordvec, sequence_length=encoder_input_seq_lengths, dtype=tf.float32)

            with tf.Variable_scope('decoder'):
                rnn_cell_list = [self._rnn_cell(ENCODER_RNN_SIZE) for _ in range(ENCODER_RNN_LAYERS_N)]
                decoder_rnn = tf.nn.rnn_cell.MultiRNNCell(rnn_cell_list)
                # Need seq2seq api

            # Get inference_logits
            # Get train_op
            test = tf.Variable(True)
        
            saver = tf.train.Saver(max_to_keep=1)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                # Train the model
                print(sess.run(test))

                # Save model at the end of the training
                saver.save(sess, self.model_ckpt_path)
                



    def predict(self, encode_str):
        return 'test'

    def save(self, path):
        pass

    def load(self, path):
        return None


if __name__ == '__main__':
    model = Seq2seq()
    encode_file_path = './data/letters_source.txt'
    decode_file_path = './data/letters_target.txt'
    encode_str = 'example'
    model.train_new(encode_file_path, decode_file_path)
    prediction_str = model.predict(encode_str)
    print("> {}".format(prediction_str))
