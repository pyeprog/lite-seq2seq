import tensorflow as tf
import numpy as np
import _pickle as pkl
import os
import re
from collections import Counter
from random import random
from multiprocessing import Pool

# GatherTree ops hasn't happened. Adding import to force library to load
# Fixed the KeyError: GatherTree
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops


# Specify save path of models
MODEL_PATH = './models'
if not os.path.isdir(MODEL_PATH):
    os.mkdir(MODEL_PATH)

# Hyper params
EMBEDDING_DIM = 300
ENCODER_RNN_SIZE = 256 # even number only
ENCODER_RNN_LAYERS_N = 3
BEAM_WIDTH = 5
DROPOUT_KEEP_PROB = 0.8
VALID_PORTION = 0.1
T_BATCH_SIZE = 128
I_BATCH_SIZE = 2
MAX_GRADIENT_NORM = 5.0
EPOCH = 10
MAX_G_STEP = float('inf')

VOCAB_COUNT_THRES = 2
MAX_VOCAB = 500

LEARNING_RATE = 1e-2
DECAY_RATE = 0.99
DECAY_STEP = 500

SHOW_EVERY = 5
SUMMARY_EVERY = 1
SAVE_EVERY = 5
DEBUG = 1

if DEBUG:
    from pprint import pprint

# Suppress warning log
tf.logging.set_verbosity(tf.logging.FATAL)


class Seq2seq:
    def __init__(self):
        self._id = (str(random())[2:] + str(random())[2:])[:20]
        self.model_ckpt_dir = os.path.join(MODEL_PATH, self._id)
        self.model_ckpt_path = os.path.join(self.model_ckpt_dir, 'checkpoint.ckpt')
        
        self.tp = TextProcessor()

    def __delete__(self):
        if hasattr(self, 'sess'):
            self.sess.close()


    def get_id(self):
        return self._id


    def get_ckpt_dir(self):
        return self.model_ckpt_dir


    def _get_ngrams(self, segment, max_order):
        ngram_counts = Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i:i+order])
                ngram_counts[ngram] += 1
        return ngram_counts


    def _bleu(self, gen_lists, refer_lists, max_order=4, smooth=False):
        matches_by_order = [0] * max_order
        possible_matches_by_order = [0] * max_order
        refer_length = 0
        gen_length = 0

        for (refer_list, gen_list) in zip(refer_lists, gen_lists):
            refer_length += len(refer_list)
            gen_length += len(gen_list)

            refer_ngram_counts = self._get_ngrams(refer_list, max_order)
            gen_ngram_counts = self._get_ngrams(gen_list, max_order)
            overlap = gen_ngram_counts & refer_ngram_counts

            for ngram in overlap:
                matches_by_order[len(ngram)-1] += overlap[ngram]

            for order in range(1, max_order+1):
                possible_matches = len(gen_list) - order + 1
                if possible_matches > 0:
                    possible_matches_by_order[order-1] += possible_matches

        precisions = [0] * max_order
        for i in range(0, max_order):
            if smooth:
                precisions[i] = ((matches_by_order[i] + 1.) /
                               (possible_matches_by_order[i] + 1.))
            else:
                if possible_matches_by_order[i] > 0:
                    precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
                else:
                    precisions[i] = 0.0

        if min(precisions) > 0:
            p_log_sum = sum((1. / max_order) * np.log(p) for p in precisions)
            geo_mean = np.exp(p_log_sum)
        else:
            geo_mean = 0.0

        ratio = float(gen_length) / refer_length

        if ratio > 1.0:
            bp = 1.
        else:
            bp = np.exp(1 - 1. / ratio)

        bleu = geo_mean * bp

        return bleu


    def _rnn_cell(self, rnn_size, dropout_keep_prob):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(rnn_size)
                #initializer=tf.random_uniform_initializer(-0.1, 0.1))
        return tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=dropout_keep_prob, output_keep_prob=1.0)

    def _word2vec_batch(self, seqs, vocab_size, batch_size):
        cur_batch_input = []
        cur_batch_target = []
        for seq in seqs:
            for i, w_id in enumerate(seq):
                cur_input = [0] * vocab_size
                cur_input[w_id] = 1
                if i < len(seq) - 1:
                    cur_batch_input.append(cur_input)

                    cur_target = [0] * vocab_size
                    cur_target[seq[i+1]] = 1
                    cur_batch_target.append(cur_target)

                if len(cur_batch_input) == batch_size:
                    yield cur_batch_input, cur_batch_target
                    cur_batch_input = []
                    cur_batch_target = []

                if 0 < i:
                    cur_batch_input.append(cur_input)

                    cur_target = [0] * vocab_size
                    cur_target[seq[i-1]] = 1
                    cur_batch_target.append(cur_target)
                    
                if len(cur_batch_input) == batch_size:
                    yield cur_batch_input, cur_batch_target
                    cur_batch_input = []
                    cur_batch_target = []

    def _padding_batch(self, inputs, targets, batch_size=T_BATCH_SIZE, input_padding_val=0, target_padding_val=0, forever=False):
        decoder_eos_id = self.decoder_vocab_to_int['<EOS>']
        while True:
            for i in range(0, len(targets) // batch_size):
                start_i = i * batch_size

                batch_inputs = inputs[start_i: start_i+batch_size]
                batch_targets = [line+[decoder_eos_id] for line in targets[start_i: start_i+batch_size]]

                batch_inputs_lens = [len(line) for line in batch_inputs]
                batch_targets_lens = [len(line) for line in batch_targets]

                inputs_cur_maxLen = np.max(batch_inputs_lens)
                targets_cur_maxLen = np.max(batch_targets_lens)

                padding_batch_inputs = np.array([line + [input_padding_val]*(inputs_cur_maxLen-len(line)) for line in batch_inputs])
                padding_batch_targets = np.array([line + [target_padding_val]*(targets_cur_maxLen-len(line)) for line in batch_targets])

                yield padding_batch_inputs, [inputs_cur_maxLen]*batch_size, padding_batch_targets, [targets_cur_maxLen]*batch_size
            
            if not forever:
                break

    
    def _parse_dict(self, file_path):
        with open(file_path, 'r') as fp:
            lines = fp.readlines()

        word_count = Counter()
        vocabs = set(['<PAD>', '<UNK>', '<GO>', '<EOS>'])
        n_lines = len(lines)
        for i, line in enumerate(lines):
            if i % 100 == 0 or i + 1 == n_lines:
                print('\rParsing dictionary {}/{}'.format(i+1, n_lines), end='', flush=True)
            for word in line.split():
                word = word.lower()
                word_count[word] += 1
        
        vocabs = ['<PAD>', '<UNK>', '<GO>', '<EOS>'] + [word_tuple[0] for word_tuple in word_count.most_common()[:MAX_VOCAB] if word_tuple[1] >= VOCAB_COUNT_THRES]
        int_to_vocab = {i:word for i, word in enumerate(vocabs)}
        vocab_to_int = {word:i for i, word in enumerate(vocabs)}
        print('\tFinished')
        return int_to_vocab, vocab_to_int

    
    def _parse_seq(self, file_path, vocab_to_int):
        with open(file_path, 'r') as fp:
            lines = fp.readlines()

        encoded_lines = []
        unk_id = vocab_to_int['<UNK>']
        n_lines = len(lines)
        for i, line in enumerate(lines):
            if i % 100 == 0 or i + 1 == n_lines:
                print('\rParsing sequence {}/{}'.format(i+1, n_lines), end='', flush=True)

            cur = [vocab_to_int.get(word.lower(), unk_id) for word in line.split()]
            if len(cur) > 0:
                encoded_lines.append(cur)

        print('\tFinished')
        return encoded_lines


    @staticmethod
    def unwrap_self_train(*arg, **kwarg):
        '''
         Process wrapper, since multiprocessing cannot call instance method
         You need a outer function
        '''
        return Seq2seq._train(*arg, **kwarg)

    
    def train(self, encode_file_path, decode_file_path):
        '''
        A process wrapper for train method
        If you use gpu to train the model, memory will not be released, even after closing session
        However, if the process is killed, memory will be released.
        '''
        with Pool(1) as process:
            params = (encode_file_path, decode_file_path)
            process.apply(self.unwrap_self_train, (self, *params))


    def _train(self, encode_file_path, decode_file_path):
        '''
        Train the model for the first time or retrain the model
        @encode_file_path: str, the path of the encoder training file
        @decode_file_path: str, the path of the decoder training file
        @return: None
        '''
        if not hasattr(self, 'sess'):
            # Train model from start
            print('Train new model')
            
            # Create dictionary
            self.encoder_int_to_vocab, self.encoder_vocab_to_int = self._parse_dict(encode_file_path)
            self.decoder_int_to_vocab, self.decoder_vocab_to_int = self._parse_dict(decode_file_path)

            # Create seqs
            self.encode_seqs = self._parse_seq(encode_file_path, self.encoder_vocab_to_int)
            self.decode_seqs = self._parse_seq(decode_file_path, self.decoder_vocab_to_int)

            # create placeholder
            ## why the shape is [None, None]? explain
            self.graph = tf.Graph()
            with self.graph.as_default():
                encoder_input = tf.placeholder(tf.int32, shape=[None, None], name='inputs')
                decoder_target = tf.placeholder(tf.int32, shape=[None, None], name='targets')
                decoder_input = tf.concat(
                        [tf.fill([T_BATCH_SIZE,1], self.decoder_vocab_to_int['<GO>']), 
                        tf.strided_slice(decoder_target, [0,0], [T_BATCH_SIZE,-1], [1,1])],
                        1)
                dropout_keep_prob = tf.placeholder(tf.float32, name='dropout')
                
                ## why does it need sequence length placeholder? explain
                encoder_input_seq_lengths = tf.placeholder(tf.int32, shape=[None,], name='source_lens')
                decoder_target_seq_lengths = tf.placeholder(tf.int32, shape=[None,], name='target_lens')

                # Build whole model and Get training_logits

                ###### ENCODER ######
                encoder_wordvec = tf.contrib.layers.embed_sequence(encoder_input, len(self.encoder_int_to_vocab), EMBEDDING_DIM,
                        initializer=tf.initializers.random_uniform(-0.1,0.1))
                
                # reshape_encoder_input = tf.reshape(encoder_input, [])
                # encoder_embedding_weights = tf.Variable(tf.random_uniform([len(self.encoder_int_to_vocab), EMBEDDING_DIM], minval=-0.1, maxval=0.1), name='encoder_embed_weight')
                # encoder_embedding_bias = tf.Variable(tf.random_uniform([EMBEDDING_DIM], minval=-0.1, maxval=0.1), name='encoder_embed_bias')
                # encoder_wordvec = tf.nn.embedding_lookup(encoder_embedding_weights, encoder_input) #+ encoder_embedding_bias

                # # To use stacked uni-directional rnn encoder, open this
                # rnn_cell_list = [self._rnn_cell(ENCODER_RNN_SIZE, dropout_keep_prob) for _ in range(ENCODER_RNN_LAYERS_N)]
                # encoder_rnn = tf.nn.rnn_cell.MultiRNNCell(rnn_cell_list)
                # encoder_output, encoder_final_state = tf.nn.dynamic_rnn(encoder_rnn, encoder_wordvec, sequence_length=encoder_input_seq_lengths, dtype=tf.float32)

                # # print(encoder_output.get_shape())
                # # print(encoder_final_state)


                # To use stacked bi-directional rnn encoder, open this
                # Explain for the state concat, explain
                rnn_cell_list_forward = [self._rnn_cell(ENCODER_RNN_SIZE // 2, dropout_keep_prob) for _ in range(ENCODER_RNN_LAYERS_N)]
                rnn_cell_list_backward = [self._rnn_cell(ENCODER_RNN_SIZE // 2, dropout_keep_prob) for _ in range(ENCODER_RNN_LAYERS_N)]

                encoder_output, forward_final_state, backward_final_state = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                        rnn_cell_list_forward, rnn_cell_list_backward, encoder_wordvec,
                        sequence_length=encoder_input_seq_lengths, time_major=False,
                        dtype=tf.float32
                        )
                

                # maybe the encoder_final_state can be updated
                # Use tensorboard to check it
                encoder_final_state = []
                for forward_cell_state, backward_cell_state in zip(forward_final_state, backward_final_state):
                    concated_state = tf.concat([forward_cell_state.c, backward_cell_state.c], -1)
                    concated_output = tf.concat([forward_cell_state.h, backward_cell_state.h], -1)
                    encoder_final_state.append(tf.nn.rnn_cell.LSTMStateTuple(concated_state, concated_output))
                encoder_final_state = tuple(encoder_final_state)

                if DEBUG:
                    tf.summary.histogram('encoder_output', encoder_output)
                    tf.summary.histogram('encoder_forward_state', forward_final_state)
                    tf.summary.histogram('encoder_backward_state', backward_final_state)




                ##### DECODER ######

                decoder_embedding_weights = tf.Variable(tf.random_uniform([len(self.decoder_int_to_vocab), EMBEDDING_DIM], minval=-0.1, maxval=0.1), name='decoder_embed_weight')
                # decoder_embedding_bias = tf.Variable(tf.random_uniform([EMBEDDING_DIM], minval=-0.1, maxval=0.1), name='decoder_embed_bias')
                decoder_wordvec = tf.nn.embedding_lookup(decoder_embedding_weights, decoder_input) #+ decoder_embedding_bias
                rnn_cell_list = [self._rnn_cell(ENCODER_RNN_SIZE, dropout_keep_prob) for _ in range(ENCODER_RNN_LAYERS_N)]
                decoder_rnn = tf.nn.rnn_cell.MultiRNNCell(rnn_cell_list)
                decoder_output_dense_layer = tf.layers.Dense(len(self.decoder_int_to_vocab), use_bias=False,
                        kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), name='decoder_output_embedding')

            # with tf.variable_scope('decoder'):
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
                        # decoder_rnn, # Used for vanilla case
                        training_helper,
                        training_decoder.zero_state(T_BATCH_SIZE,tf.float32).clone(cell_state=encoder_final_state),
                        # encoder_final_state, # Used for vanilla case
                        decoder_output_dense_layer
                        )

                training_decoder_output = tf.contrib.seq2seq.dynamic_decode(
                        training_decoder,
                        impute_finished=True,
                        maximum_iterations=tf.reduce_max(decoder_target_seq_lengths)
                        )[0]

            # with tf.variable_scope('decoder', reuse=True):
                # Tiled start_token <GO>
                start_tokens = tf.tile(
                        tf.constant([self.decoder_vocab_to_int['<GO>']], dtype=tf.int32),
                        [I_BATCH_SIZE],
                        name='start_tokens')

                # # To use greedy decoder, open this
                # inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                 #     decoder_embedding_weights,
                 #     start_tokens,
                 #     self.decoder_vocab_to_int['<EOS>']
                 #     )

                # inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                 #     inference_decoder,
                 #     # decoder_rnn, # Used for vanilla case
                 #     inference_helper,
                 #     inference_decoder.zero_state(T_BATCH_SIZE,tf.float32).clone(cell_state=encoder_final_state),
                 #     # encoder_final_state, # Used for vanilla case
                 #     decoder_output_dense_layer
                 #     )

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
                        inference_decoder.zero_state(I_BATCH_SIZE*BEAM_WIDTH,tf.float32).clone(
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



                ##### OPTIMIZATION #####

                # Get train_op
                training_logits = tf.identity(training_decoder_output.rnn_output, name='logits')
                inference_logits = tf.identity(inference_decoder_output.predicted_ids[:,:,0], name='predictions')
                # inference_logits = tf.identity(inference_decoder_output.rnn_output, name='predictions')
                decoder_output = tf.identity(training_decoder_output.sample_id, name='training_output')

                # Why mask, explain
                mask = tf.sequence_mask(decoder_target_seq_lengths, tf.reduce_max(decoder_target_seq_lengths), dtype=tf.float32, name='mask')
                
                global_step = tf.Variable(0, trainable=False)

                # Cost
                cost = tf.contrib.seq2seq.sequence_loss(
                        training_logits,
                        decoder_target,
                        mask,
                        name='cost'
                        )

                # # Cost alternative
                # crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                #         labels=decoder_target, logits=training_logits
                #         )
                # cost = (tf.reduce_sum(crossent * mask) / T_BATCH_SIZE)


                # lr = tf.train.exponential_decay(LEARNING_RATE, global_step, DECAY_STEP, DECAY_RATE, True)
                # optimizer = tf.train.AdamOptimizer(lr)
                # gradients = optimizer.compute_gradients(cost)
                # capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
                # train_op = optimizer.apply_gradients(capped_gradients, global_step=global_step, name='train_op')


                # Clip by global norm
                trainable_params = tf.trainable_variables()
                gradients = tf.gradients(cost, trainable_params)
                capped_gradients,_ = tf.clip_by_global_norm(gradients, MAX_GRADIENT_NORM)
                optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
                train_op = optimizer.apply_gradients(zip(capped_gradients, trainable_params), global_step=global_step, name='train_op')

                if DEBUG:
                    tf.summary.scalar('seq_loss', cost)
                    tf.summary.scalar('learning_rate', optimizer._lr)

                    trainable_params = tf.trainable_variables()
                    gradients = tf.gradients(cost, trainable_params)
                    for param, gradient in zip(trainable_params, gradients):
                        if gradient is not None:
                            tf.summary.histogram(param.name, gradient)


                # Save op to collection for further use
                tf.add_to_collection("optimization", train_op)
                tf.add_to_collection("optimization", cost)
                tf.add_to_collection("optimization", global_step)

                # Initialize the graph variables
                self.sess = tf.Session()
                self.sess.run(tf.global_variables_initializer())

                # Save dictionary
                if not os.path.isdir(self.model_ckpt_dir):
                    os.mkdir(self.model_ckpt_dir)

                with open(os.path.join(self.model_ckpt_dir, 'dictionary'), 'wb') as fp:
                    dictionary = (self.encoder_int_to_vocab, self.encoder_vocab_to_int, self.decoder_int_to_vocab, self.decoder_vocab_to_int)
                    pkl.dump(dictionary, fp)

        else:
            # Pre-trained model has loaded
            print('Load pre-trained model')

            # Create seqs
            self.encode_seqs = self._parse_seq(encode_file_path, self.encoder_vocab_to_int)
            self.decode_seqs = self._parse_seq(decode_file_path, self.decoder_vocab_to_int)

            with self.graph.as_default():
                encoder_input = self.graph.get_tensor_by_name('inputs:0')
                encoder_input_seq_lengths = self.graph.get_tensor_by_name('source_lens:0')
                decoder_target = self.graph.get_tensor_by_name('targets:0')
                decoder_target_seq_lengths = self.graph.get_tensor_by_name('target_lens:0')
                dropout_keep_prob = self.graph.get_tensor_by_name('dropout:0')
                decoder_output = self.graph.get_tensor_by_name('optimization/training_output:0')
                train_op = tf.get_collection("optimization")[0]
                cost = tf.get_collection("optimization")[1]
                global_step = tf.get_collection("optimization")[2]


        encode_pad_id = self.encoder_vocab_to_int['<PAD>']
        decode_pad_id = self.decoder_vocab_to_int['<PAD>']

        # Validate set reserve
        n_valid_data = int(len(self.encode_seqs) * VALID_PORTION) // T_BATCH_SIZE * T_BATCH_SIZE
        valid_encode_seqs = self.encode_seqs[:n_valid_data]
        valid_decode_seqs = self.decode_seqs[:n_valid_data]
        valid_batch_generator = self._padding_batch(valid_encode_seqs, valid_decode_seqs, T_BATCH_SIZE, encode_pad_id, decode_pad_id, forever=True)

        train_encode_seqs = self.encode_seqs[n_valid_data:]
        train_decode_seqs = self.decode_seqs[n_valid_data:]

        n_batch = len(train_encode_seqs) // T_BATCH_SIZE

        # Train the model
        with self.graph.as_default():
            # Create a saver
            saver = tf.train.Saver(max_to_keep=1)

            if DEBUG:
                summary_writer = tf.summary.FileWriter(os.path.join(self.model_ckpt_dir, 'tensorboard'))
                summary_writer.add_graph(self.sess.graph)
                summary_ops = tf.summary.merge_all()

            g_step = self.sess.run(global_step) % n_batch
            # Pass trained batch
            batch_generator = self._padding_batch(train_encode_seqs, train_decode_seqs, T_BATCH_SIZE, encode_pad_id, decode_pad_id)
            for _ in range(g_step):
                _ = next(batch_generator)

            # Start training
            for epoch_i in range(1, EPOCH+1):
                for cur_batch_pack in batch_generator:
                    inputs, inputs_lens, targets, targets_lens = cur_batch_pack

                    # print(inputs)
                    # print(inputs_lens)
                    
                    _, train_loss, g_step = self.sess.run(
                            [train_op, cost, global_step],
                            feed_dict={
                                encoder_input:inputs,
                                encoder_input_seq_lengths:inputs_lens,
                                decoder_target:targets,
                                decoder_target_seq_lengths:targets_lens,
                                dropout_keep_prob: DROPOUT_KEEP_PROB
                                }
                            )
                    print("\r{}/{} ".format(g_step % n_batch, n_batch), end='', flush=True)

                    if g_step % SHOW_EVERY == 0:
                        valid_batch_pack = next(valid_batch_generator)
                        valid_inputs, valid_inputs_lens, valid_targets, valid_targets_lens = valid_batch_pack

                        val_loss, prediction_lists = self.sess.run([cost, decoder_output], feed_dict={
                            encoder_input:valid_inputs,
                            encoder_input_seq_lengths:valid_inputs_lens,
                            decoder_target:valid_targets,
                            decoder_target_seq_lengths:valid_targets_lens,
                            dropout_keep_prob: 1.0
                            })

                        bleu_score = self._bleu(prediction_lists, valid_targets)
                        print("E:{}/{} B:{} - train loss: {}\tvalid loss: {}\tvalid bleu: {}".format(epoch_i, EPOCH, g_step, train_loss, val_loss, bleu_score))
                    
                    if g_step % SAVE_EVERY == 0:
                        saver.save(self.sess, self.model_ckpt_path, write_meta_graph=(g_step==0))

                    if DEBUG and g_step % SUMMARY_EVERY == 0:
                        summary_info = self.sess.run(summary_ops, feed_dict={
                            encoder_input:inputs,
                            encoder_input_seq_lengths:inputs_lens,
                            decoder_target:targets,
                            decoder_target_seq_lengths:targets_lens,
                            dropout_keep_prob: DROPOUT_KEEP_PROB
                            })
                        summary_writer.add_summary(summary_info, self.sess.run(global_step))

                        # weights = self.sess.run([encoder_embedding_weights, decoder_embedding_weights])
                        # wordvec = self.sess.run([encoder_wordvec, decoder_wordvec], feed_dict={
                        #     encoder_input:inputs,
                        #     encoder_input_seq_lengths:inputs_lens,
                        #     decoder_target:targets,
                        #     decoder_target_seq_lengths:targets_lens,
                        #     dropout_keep_prob: DROPOUT_KEEP_PROB
                        #     })
                        # if g_step > 20:
                        #     print(wordvec[0])
                        #     # pprint(wordvec[1])


                    if g_step > MAX_G_STEP:
                        break
                if g_step > MAX_G_STEP:
                    break
                
                # Get a new batch-generator
                batch_generator = self._padding_batch(train_encode_seqs, train_decode_seqs, T_BATCH_SIZE, encode_pad_id, decode_pad_id)


    def predict(self, encode_str):
        if not hasattr(self, 'sess'):
            self.load(self.model_ckpt_dir)

        encoder_unk_id = self.encoder_vocab_to_int['<UNK>']
        decoder_pad_id = self.decoder_vocab_to_int['<PAD>']
        decoder_eos_id = self.decoder_vocab_to_int['<EOS>']

        # Parse encode_str
        encode_str = self.tp.process_str(encode_str)
        inputs = [[self.encoder_vocab_to_int.get(word, encoder_unk_id) for word in encode_str.split()]]
        inputs_lens = [len(line) for line in inputs]

        with self.graph.as_default():
            encoder_input = self.graph.get_tensor_by_name('inputs:0')
            encoder_input_seq_lengths = self.graph.get_tensor_by_name('source_lens:0')
            decoder_target_seq_lengths = self.graph.get_tensor_by_name('target_lens:0')
            dropout_keep_prob = self.graph.get_tensor_by_name('dropout:0')
            prediction = self.graph.get_tensor_by_name('optimization/predictions:0')

            predict_list = self.sess.run(
                    prediction, 
                    feed_dict={
                        encoder_input:inputs*I_BATCH_SIZE,
                        encoder_input_seq_lengths:inputs_lens*I_BATCH_SIZE,
                        dropout_keep_prob: 1.0
                        }
                    )

        return ' '.join([self.decoder_int_to_vocab.get(i, '') for i in predict_list[0]])# if i!=decoder_pad_id and i!=decoder_eos_id])
        # print(predict_list)


    def load(self, path):
        if not os.path.isdir(path):
            raise ValueError('{} is not valid path, your model is probably untrained'.format(path))

        self._id = os.path.basename(path)
        self.model_ckpt_dir = path
        self.model_ckpt_path = os.path.join(path, 'checkpoint.ckpt')
        
        if not os.path.isfile(os.path.join(path, 'checkpoint')):
            raise ValueError('There is no checkpoint file in {}, your model has not finished training'.format(path))

        if not os.path.isfile(os.path.join(path, 'dictionary')):
            raise ValueError('There is no dictionary file in {}'.format(path))

        with open(os.path.join(path, 'dictionary'), 'rb') as fp:
            self.encoder_int_to_vocab, self.encoder_vocab_to_int, self.decoder_int_to_vocab, self.decoder_vocab_to_int = pkl.load(fp)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            loader = tf.train.import_meta_graph(self.model_ckpt_path+'.meta')
            loader.restore(self.sess, tf.train.latest_checkpoint(path))


class TextProcessor:
    def __init__(self):
        self.proc_fn_list = [self.proc1, self.proc2, self.proc3, self.proc4, self.proc5, self.proc6, self.proc7, self.proc8]

    # pickle cann't dump lambda function in py3, so...
    def proc1(self, x):
        return re.sub('\(.*?\)', '', x)
    def proc2(self, x):
        return re.sub('\[.*?\]', '', x)
    def proc3(self, x):
        return re.sub('\{.*?\}', '', x)
    def proc4(self, x):
        return re.sub('\w\.{,1}\w\.*', lambda y:y.group().replace('.',''), x)
    def proc5(self, x):
        return re.sub('[:\-\/\\*&$#@\^]+|\.{2,}', ' ', x)
    def proc6(self, x):
        return re.sub('[,.!?;]+', lambda y:' '+y.group()+' ', x)
    def proc7(self, x):
        return re.sub('[\=\<\>\"\`\(\)\[\]\{\}]+', '', x)
    def proc8(self, x):
        return re.sub('\'+s*', '', x)

    def read(self, file_path):
        if not os.path.isfile(file_path):
            raise ValueError('{} is not valid file path'.format(file_path))
        else:
            self.file_path = file_path

        with open(file_path, 'r') as fp:
            self.lines = fp.readlines()

        return self

    def process(self, proc_fn_list=[], inplace=False):
        if len(proc_fn_list) == 0:
            proc_fn_list = self.proc_fn_list

        new_lines = []
        n_lines = len(self.lines)
        for i, line in enumerate(self.lines):
            if i % 1000 == 0: print('\rProcessing {}/{}'.format(i+1, n_lines), end='', flush=True)
            for fn in proc_fn_list:
                line = fn(line)
            line += '\n' if line[-1] != '\n' else ''
            new_lines.append(line)

        new_content = ''.join(new_lines)

        if not inplace:
            return new_lines

        else:
            with open(self.file_path+'.proc', 'w') as fp:
                fp.write(new_content)
            return self.file_path+'.proc'

    def process_str(self, string, proc_fn_list=[]):
        if len(proc_fn_list) == 0:
            proc_fn_list = self.proc_fn_list

        for fn in proc_fn_list:
            string = fn(string)

        return string



if __name__ == '__main__':
    model = Seq2seq()
    encode_file_path = './data/en_fr/small_vocab_en'
    decode_file_path = './data/en_fr/small_vocab_fr'

    # model.load('./models/94193666769558898599')
    model._train(encode_file_path, decode_file_path)
    while True:
        encode_str = input('< ')
        print('>> {}'.format(model.predict(encode_str)))

    # file1_path = '/Users/pd/Downloads/europarl-v7.fr-en.en'
    # file2_path = '/Users/pd/Downloads/europarl-v7.fr-en.fr'

    # tp = TextProcessor()
    # new_lines = tp.read(file1_path).process()
    # print(new_lines)
    # # tp.read(file2_path).process()
