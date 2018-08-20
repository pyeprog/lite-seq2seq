import tensorflow as tf
import numpy as np
import _pickle as pkl
import os
from random import random

MODEL_PATH = './models'
if not os.path.isdir(MODEL_PATH):
    os.mkdir(MODEL_PATH)

EMBEDDING_DIM = 128
ENCODER_RNN_SIZE = 256
ENCODER_RNN_LAYERS_N = 3
BEAM_WIDTH = 5
BATCH_SIZE = 32
LEARNING_RATE = 7e-4
EPOCH = 1

class Seq2seq:
    def __init__(self):
        self._id = (str(random())[2:] + str(random())[2:])[:20]
        self.model_ckpt_dir = os.path.join(MODEL_PATH, self._id)
        self.model_ckpt_path = os.path.join(self.model_ckpt_dir, 'checkpoint.ckpt')

    def __delete__(self):
        if hasattr(self, 'sess'):
            self.sess.close()

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
        decoder_eos_id = self.decoder_vocab_to_int['<EOS>']
        for i in range(0, len(targets) // batch_size):
            start_i = i * batch_size
            batch_inputs = inputs[start_i: start_i+batch_size]
            batch_targets = [line+[decoder_eos_id] for line in targets[start_i: start_i+batch_size]]

            batch_inputs_lens = [len(line) for line in batch_inputs]
            batch_targets_lens = [len(line) for line in batch_targets]

            inputs_cur_maxLen = max(batch_inputs_lens)
            targets_cur_maxLen = max(batch_targets_lens)

            padding_batch_inputs = np.array([line + [input_padding_val]*(inputs_cur_maxLen-len(line)) for line in batch_inputs])
            padding_batch_targets = np.array([line + [target_padding_val]*(targets_cur_maxLen-len(line)) for line in batch_targets])

            yield padding_batch_inputs, batch_inputs_lens, padding_batch_targets, batch_targets_lens


    def _init_data(self):
        self.encode_seqs, self.encoder_int_to_vocab, self.encoder_vocab_to_int = self._text_file_parse(encode_file_path)
        self.decode_seqs, self.decoder_int_to_vocab, self.decoder_vocab_to_int = self._text_file_parse(decode_file_path)


    def train(self, encode_file_path, decode_file_path):
        '''
        Train the model for the first time or retrain the model
        @encode_file_path: str, the path of the encoder training file
        @decode_file_path: str, the path of the decoder training file
        @return: None
        '''
        self._init_data()
        if not hasattr(self, 'sess'):
            # Train model from start
            self.graph = tf.Graph()
            with self.graph.as_default():
                # create placeholder
                ## why the shape is [None, None]? explain
                encoder_input = tf.placeholder(tf.int32, shape=[None, None], name='inputs')
                decoder_target = tf.placeholder(tf.int32, shape=[None, None], name='targets')
                decoder_input = tf.concat(
                        [tf.fill([BATCH_SIZE,1], self.decoder_vocab_to_int['<GO>']), 
                        tf.strided_slice(decoder_target, [0,0], [BATCH_SIZE,-1], [1,1])],
                        1)
                
                ## why does it need sequence length placeholder? explain
                encoder_input_seq_lengths = tf.placeholder(tf.int32, shape=[None,], name='source_lens')
                decoder_target_seq_lengths = tf.placeholder(tf.int32, shape=[None,], name='target_lens')

                # Build whole model and Get training_logits
                with tf.variable_scope('encoder'):
                    encoder_wordvec = tf.contrib.layers.embed_sequence(encoder_input, len(self.encoder_int_to_vocab), EMBEDDING_DIM)

                    # # To use stacked uni-directional rnn encoder, open this
                    # rnn_cell_list = [self._rnn_cell(ENCODER_RNN_SIZE) for _ in range(ENCODER_RNN_LAYERS_N)]
                    # encoder_rnn = tf.nn.rnn_cell.MultiRNNCell(rnn_cell_list)
                    # encoder_output, encoder_final_state = tf.nn.dynamic_rnn(encoder_rnn, encoder_wordvec, sequence_length=encoder_input_seq_lengths, dtype=tf.float32)
                    # print(encoder_output.get_shape())
                    # print(encoder_final_state)


                    # To use stacked bi-directional rnn encoder, open this
                    # Explain for the state concat, explain
                    rnn_cell_list_forward = [self._rnn_cell(ENCODER_RNN_SIZE // 2) for _ in range(ENCODER_RNN_LAYERS_N)]
                    rnn_cell_list_backward = [self._rnn_cell(ENCODER_RNN_SIZE // 2) for _ in range(ENCODER_RNN_LAYERS_N)]

                    encoder_output, forward_final_state, backward_final_state = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                            rnn_cell_list_forward, rnn_cell_list_backward, encoder_wordvec,
                            sequence_length=encoder_input_seq_lengths, time_major=False,
                            dtype=tf.float32
                            )

                    encoder_final_state = []
                    for forward_cell_state, backward_cell_state in zip(forward_final_state, backward_final_state):
                        concated_state = tf.concat([forward_cell_state.c, backward_cell_state.c], -1)
                        concated_output = tf.concat([forward_cell_state.h, backward_cell_state.h], -1)
                        encoder_final_state.append(tf.nn.rnn_cell.LSTMStateTuple(concated_state, concated_output))
                    encoder_final_state = tuple(encoder_final_state)



                with tf.variable_scope('decoder_prepare'):
                    decoder_embedding_weights = tf.Variable(tf.random_uniform([len(self.decoder_int_to_vocab), EMBEDDING_DIM]))
                    decoder_wordvec = tf.nn.embedding_lookup(decoder_embedding_weights, decoder_input)
                    rnn_cell_list = [self._rnn_cell(ENCODER_RNN_SIZE) for _ in range(ENCODER_RNN_LAYERS_N)]
                    decoder_rnn = tf.nn.rnn_cell.MultiRNNCell(rnn_cell_list)
                    decoder_output_dense_layer = tf.layers.Dense(len(self.decoder_int_to_vocab), use_bias=False,
                            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

                with tf.variable_scope('decoder'):
                    training_helper = tf.contrib.seq2seq.TrainingHelper(
                            inputs=decoder_wordvec,
                            sequence_length=decoder_target_seq_lengths,
                            time_major=False)

                    # Add attention mechanism
                    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                            ENCODER_RNN_SIZE, encoder_output,
                            memory_sequence_length=encoder_input_seq_lengths
                            )

                    # Wrapper Attention mechanism on plain rnn cell first
                    training_decoder = tf.contrib.seq2seq.AttentionWrapper(
                            decoder_rnn, attention_mechanism,
                            attention_layer_size=ENCODER_RNN_SIZE
                            )

                    # Make decoder and it's initial state with wrapped rnn cell
                    training_decoder = tf.contrib.seq2seq.BasicDecoder(
                            training_decoder,
                            training_helper,
                            training_decoder.zero_state(BATCH_SIZE,tf.float32).clone(cell_state=encoder_final_state),
                            decoder_output_dense_layer
                            )

                    training_decoder_output = tf.contrib.seq2seq.dynamic_decode(
                            training_decoder,
                            impute_finished=True,
                            maximum_iterations=tf.reduce_max(decoder_target_seq_lengths)
                            )[0]

                with tf.variable_scope('decoder', reuse=True):
                    # Tiled start_token <GO>
                    start_tokens = tf.tile(
                            tf.constant([self.decoder_vocab_to_int['<GO>']], dtype=tf.int32),
                            [BATCH_SIZE],
                            name='start_tokens')

                    # # To use greedy decoder, open this
                    # inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    #         decoder_embedding_weights,
                    #         start_tokens,
                    #         self.decoder_vocab_to_int['<EOS>']
                    #         )
                    #
                    # inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                    #         inference_decoder,
                    #         inference_helper,
                    #         inference_decoder.zero_state(BATCH_SIZE,tf.float32).clone(cell_state=encoder_final_state),
                    #         decoder_output_dense_layer
                    #         )

                    # To use beam search decoder, open this
                    # Beam search tile
                    tiled_encoder_output = tf.contrib.seq2seq.tile_batch(encoder_output, multiplier=BEAM_WIDTH)
                    tiled_encoder_input_seq_lengths = tf.contrib.seq2seq.tile_batch(encoder_input_seq_lengths, multiplier=BEAM_WIDTH)
                    # Explain the tile state, need explain, tile_batch can handle nested state
                    tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(encoder_final_state, multiplier=BEAM_WIDTH)

                    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                            ENCODER_RNN_SIZE, tiled_encoder_output,
                            memory_sequence_length=tiled_encoder_input_seq_lengths
                            )

                    inference_decoder = tf.contrib.seq2seq.AttentionWrapper(
                            decoder_rnn, attention_mechanism,
                            attention_layer_size=ENCODER_RNN_SIZE
                            )

                    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                            inference_decoder,
                            decoder_embedding_weights,
                            start_tokens,
                            self.decoder_vocab_to_int['<EOS>'],
                            inference_decoder.zero_state(BATCH_SIZE*BEAM_WIDTH,tf.float32).clone(
                                cell_state=tiled_encoder_final_state
                                ),
                            BEAM_WIDTH,
                            decoder_output_dense_layer,
                            length_penalty_weight=0.0
                            )

                    inference_decoder_output = tf.contrib.seq2seq.dynamic_decode(
                            inference_decoder,
                            impute_finished=False,
                            maximum_iterations=2*tf.reduce_max(encoder_input_seq_lengths)
                            )[0]

                # Get train_op
                with tf.variable_scope('optimization'):
                    training_logits = tf.identity(training_decoder_output.rnn_output, name='logits')
                    inference_logits = tf.identity(inference_decoder_output.predicted_ids[:,:,0], name='predictions')

                    # Why mask, explain
                    mask = tf.sequence_mask(decoder_target_seq_lengths, tf.reduce_max(decoder_target_seq_lengths), dtype=tf.float32, name='mask')
                    cost = tf.contrib.seq2seq.sequence_loss(
                            training_logits,
                            decoder_target,
                            mask,
                            name='cost'
                            )
                    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
                    gradients = optimizer.compute_gradients(cost)
                    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
                    train_op = optimizer.apply_gradients(capped_gradients, name='train_op')

                self.sess = tf.Session()
                self.sess.run(tf.global_variables_initializer())

        else:
            # Pre-trained model has loaded
            with self.graph.as_default():
                encoder_input = self.graph.get_tensor_by_name('inputs:0')
                encoder_input_seq_lengths = self.graph.get_tensor_by_name('source_lens:0')
                decoder_target = self.graph.get_tensor_by_name('targets:0')
                decoder_target_seq_lengths = self.graph.get_tensor_by_name('target_lens:0')
                train_op = tf.get_collection("optimization")[0]
                cost = tf.get_collection("optimization")[1]

        with self.graph.as_default():

            encode_pad_id = self.encoder_vocab_to_int['<PAD>']
            decode_pad_id = self.decoder_vocab_to_int['<PAD>']

            # Train the model
            for epoch_i in range(1, EPOCH+1):
                batch_i = 0
                for cur_batch_pack in self._padding_batch(self.encode_seqs, self.decode_seqs, BATCH_SIZE, encode_pad_id, decode_pad_id):
                    batch_i += 1
                    inputs, inputs_lens, targets, targets_lens = cur_batch_pack
                    
                    _, loss = self.sess.run(
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
            if not os.path.isdir(self.model_ckpt_dir):
                os.mkdir(self.model_ckpt_dir)

            tf.add_to_collection("optimization", train_op)
            tf.add_to_collection("optimization", cost)

            saver = tf.train.Saver(max_to_keep=1)
            saver.save(self.sess, self.model_ckpt_path)
            with open(os.path.join(self.model_ckpt_dir, 'dictionary'), 'wb') as fp:
                dictionary = (self.encoder_int_to_vocab, self.encoder_vocab_to_int, self.decoder_int_to_vocab, self.decoder_vocab_to_int)
                pkl.dump(dictionary, fp)
            print("Model has been trained and saved")



    def predict(self, encode_str):
        if not hasattr(self, 'sess'):
            raise RuntimeError('Your model is untrained, please train it first')

        encoder_unk_id = self.encoder_vocab_to_int['<UNK>']
        decoder_pad_id = self.decoder_vocab_to_int['<PAD>']
        decoder_eos_id = self.decoder_vocab_to_int['<EOS>']

        # Parse encode_str
        encode_str = encode_str.replace('\n', ' ')
        inputs = [[self.encoder_vocab_to_int.get(word, encoder_unk_id) for word in encode_str.split()]]
        inputs_lens = [len(line) for line in inputs]

        encoder_input = self.graph.get_tensor_by_name('inputs:0')
        encoder_input_seq_lengths = self.graph.get_tensor_by_name('source_lens:0')
        decoder_target_seq_lengths = self.graph.get_tensor_by_name('target_lens:0')
        prediction = self.graph.get_tensor_by_name('optimization/predictions:0')

        predict_list = self.sess.run(
                prediction, 
                feed_dict={
                    encoder_input:inputs*BATCH_SIZE,
                    encoder_input_seq_lengths:inputs_lens*BATCH_SIZE,
                    }
                )
        return ' '.join([self.decoder_int_to_vocab.get(i, '') for i in predict_list[0] if i!=decoder_pad_id and i!=decoder_eos_id])


    def load(self, path):
        self._id = os.path.basename(path)
        self.model_ckpt_path = os.path.join(path, 'checkpoint.ckpt')
        with open(os.path.join(path, 'dictionary'), 'rb') as fp:
            self.encoder_int_to_vocab, self.encoder_vocab_to_int, self.decoder_int_to_vocab, self.decoder_vocab_to_int = pkl.load(fp)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph)
            loader = tf.train.import_meta_graph(self.model_ckpt_path+'.meta')
            loader.restore(self.sess, tf.train.latest_checkpoint(path))

if __name__ == '__main__':
    model = Seq2seq()
    encode_file_path = './data/test_en_fr/test_enc.txt'
    decode_file_path = './data/test_en_fr/test_dec.txt'
    # model.load('./models/14877418162848366253')
    model.train(encode_file_path, decode_file_path)
    while True:
        encode_str = input('< ')
        print('>> {}'.format(model.predict(encode_str)))

    # prediction_str = model.predict(encode_str)
    # print("> {}".format(prediction_str))


    # Test _text_file_parse
    # test_content, int_to_vocab = model._text_file_parse(encode_file_path)
    # print([len(line) for line in test_content])
    # print(int_to_vocab)
