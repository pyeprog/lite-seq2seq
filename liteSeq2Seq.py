import tensorflow as tf
import numpy as np
import os
from random import random

MODEL_PATH = './models'
if not os.path.isdir(MODEL_PATH):
    os.mkdir(MODEL_PATH)

EMBEDDING_DIM = 128
ENCODER_RNN_SIZE = 256
ENCODER_RNN_LAYERS_N = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCH = 20

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
        int_contents = [[vocab_to_int.get(word, unk_word_id) for word in line.split()] for line in text_contents if len(line.split()) > 0]
        return int_contents, int_to_vocab, vocab_to_int


    def _rnn_cell(self, rnn_size):
        return tf.nn.rnn_cell.LSTMCell(rnn_size,
                initializer=tf.random_uniform_initializer(-0.1, 0.1))

    def _padding_batch(self, inputs, targets, batch_size=BATCH_SIZE, input_padding_val=0, target_padding_val=0):
        for i in range(0, len(targets) // batch_size):
            start_i = i * batch_size
            batch_inputs = inputs[start_i: start_i+batch_size]
            batch_targets = targets[start_i: start_i+batch_size]

            batch_inputs_lens = [len(line) for line in batch_inputs]
            batch_targets_lens = [len(line) for line in batch_targets]

            inputs_cur_maxLen = max(batch_inputs_lens)
            targets_cur_maxLen = max(batch_targets_lens)

            padding_batch_inputs = np.array([line + [input_padding_val]*(inputs_cur_maxLen-len(line)) for line in batch_inputs])
            padding_batch_targets = np.array([line + [target_padding_val]*(targets_cur_maxLen-len(line)) for line in batch_targets])

            yield padding_batch_inputs, batch_inputs_lens, padding_batch_targets, batch_targets_lens

    def train(self, encode_file_path, decode_file_path):
        pass


    def train_new(self, encode_file_path, decode_file_path):
        '''
        Train the model for the first time or retrain the model
        @encode_file_path: str, the path of the encoder training file
        @decode_file_path: str, the path of the decoder training file
        @return: None
        '''
        encode_seqs, encoder_int_to_vocab, encoder_vocab_to_int = self._text_file_parse(encode_file_path)
        decode_seqs, decoder_int_to_vocab, decoder_vocab_to_int = self._text_file_parse(decode_file_path)

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
            with tf.variable_scope('encoder'):
                encoder_wordvec = tf.contrib.layers.embed_sequence(encoder_input, len(encoder_int_to_vocab), EMBEDDING_DIM)
                rnn_cell_list = [self._rnn_cell(ENCODER_RNN_SIZE) for _ in range(ENCODER_RNN_LAYERS_N)]
                encoder_rnn = tf.nn.rnn_cell.MultiRNNCell(rnn_cell_list)
                encoder_output, encoder_final_state = tf.nn.dynamic_rnn(encoder_rnn, encoder_wordvec, sequence_length=encoder_input_seq_lengths, dtype=tf.float32)

            with tf.variable_scope('decoder_prepare'):
                decoder_embedding_weights = tf.Variable(tf.random_uniform([len(decoder_int_to_vocab), EMBEDDING_DIM]))
                decoder_wordvec = tf.nn.embedding_lookup(decoder_embedding_weights, decoder_target)
                rnn_cell_list = [self._rnn_cell(ENCODER_RNN_SIZE) for _ in range(ENCODER_RNN_LAYERS_N)]
                decoder_rnn = tf.nn.rnn_cell.MultiRNNCell(rnn_cell_list)
                decoder_output_dense_layer = tf.layers.Dense(len(decoder_int_to_vocab),
                        kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            with tf.variable_scope('decoder'):
                training_helper = tf.contrib.seq2seq.TrainingHelper(
                        inputs=decoder_wordvec,
                        sequence_length=decoder_target_seq_lengths,
                        time_major=False)
                training_decoder = tf.contrib.seq2seq.BasicDecoder(
                        decoder_rnn,
                        training_helper,
                        encoder_final_state,
                        decoder_output_dense_layer
                        )
                training_decoder_output = tf.contrib.seq2seq.dynamic_decode(
                        training_decoder,
                        impute_finished=True,
                        maximum_iterations=len(decoder_int_to_vocab)
                        )[0]

            with tf.variable_scope('decoder', reuse=True):
                start_tokens = tf.tile(
                        tf.constant([decoder_vocab_to_int['<GO>']], dtype=tf.int32),
                        [BATCH_SIZE],
                        name='start_tokens')
                inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        decoder_embedding_weights,
                        start_tokens,
                        decoder_vocab_to_int['<EOS>']
                        )
                inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                        decoder_rnn,
                        inference_helper,
                        encoder_final_state,
                        decoder_output_dense_layer
                        )
                inference_decoder_output = tf.contrib.seq2seq.dynamic_decode(
                        inference_decoder,
                        impute_finished=True,
                        maximum_iterations=len(decoder_int_to_vocab)
                        )[0]


            # Get train_op
            with tf.variable_scope('optimization'):
                training_logits = tf.identity(training_decoder_output.rnn_output, name='logits')
                inference_logits = tf.identity(inference_decoder_output.sample_id, name='prediction')

                # Why mask, explain
                mask = tf.sequence_mask(decoder_target_seq_lengths, tf.reduce_max(decoder_target_seq_lengths), dtype=tf.float32, name='mask')
                cost = tf.contrib.seq2seq.sequence_loss(
                        training_logits,
                        decoder_target,
                        mask
                        )
                optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
                gradients = optimizer.compute_gradients(cost)
                capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
                train_op = optimizer.apply_gradients(capped_gradients)
        
            saver = tf.train.Saver(max_to_keep=1)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                encode_pad_val = encoder_vocab_to_int['<PAD>']
                decode_pad_val = decoder_vocab_to_int['<PAD>']

                # Train the model
                for epoch_i in range(1, EPOCH+1):
                    batch_i = 0
                    for cur_batch_pack in self._padding_batch(encode_seqs, decode_seqs, BATCH_SIZE, encode_pad_val, decode_pad_val):
                        batch_i += 1
                        inputs, inputs_lens, targets, targets_lens = cur_batch_pack
                        
                        _, loss = sess.run(
                                [train_op, cost],
                                feed_dict={
                                    encoder_input:inputs,
                                    encoder_input_seq_lengths:inputs_lens,
                                    decoder_target:targets,
                                    decoder_target_seq_lengths:targets_lens
                                    }
                                )

                        print("E:{}  B:{} - Loss: {}".format(epoch_i, batch_i, loss))


                # Save model at the end of the training
                saver.save(sess, self.model_ckpt_path)
                print("Model saved")
                



    def predict(self, encode_str):
        return 'test'

    def save(self, path):
        pass

    def load(self, path):
        return None


if __name__ == '__main__':
    model = Seq2seq()
    encode_file_path = './data/test_en_fr/test_enc.txt'
    decode_file_path = './data/test_en_fr/test_dec.txt'
    encode_str = 'example'
    model.train_new(encode_file_path, decode_file_path)
    # prediction_str = model.predict(encode_str)
    # print("> {}".format(prediction_str))


    # Test _text_file_parse
    # test_content, int_to_vocab = model._text_file_parse(encode_file_path)
    # print([len(line) for line in test_content])
    # print(int_to_vocab)
