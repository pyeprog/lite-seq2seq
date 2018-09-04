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
DROPOUT_KEEP_PROB = 0.5
VALID_PORTION = 0.1
T_BATCH_SIZE = 128
I_BATCH_SIZE = 2
MAX_GRADIENT_NORM = 5.0
EPOCH = 10
EMB_EPOCH = 5
EMB_BATCH = 64
MAX_G_STEP = float('inf')

VOCAB_COUNT_THRES = 2
MAX_VOCAB = 5000000

LEARNING_RATE = 1e-6
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


import copy
import pickle
CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3 }


# Number of Epochs
epochs = 6
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 256
# Number of Layers
num_layers = 3
# Embedding Size
encoding_embedding_size = 300
decoding_embedding_size = 300
# Learning Rate
learning_rate = 0.0001
# Dropout Keep Probability
keep_probability = 0.5
display_step = 20


def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, 'r', encoding='utf-8') as f:
        return f.read()


def preprocess_and_save_data(source_path, target_path, text_to_ids):
    """
    Preprocess Text Data.  Save to to file.
    """
    # Preprocess
    source_text = load_data(source_path)
    target_text = load_data(target_path)

    source_text = source_text.lower()
    target_text = target_text.lower()

    source_vocab_to_int, source_int_to_vocab = create_lookup_tables(source_text)
    target_vocab_to_int, target_int_to_vocab = create_lookup_tables(target_text)

    source_text, target_text = text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int)

    # Save Data
    with open('preprocess.p', 'wb') as out_file:
        pickle.dump((
            (source_text, target_text),
            (source_vocab_to_int, target_vocab_to_int),
            (source_int_to_vocab, target_int_to_vocab)), out_file)


def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    with open('preprocess.p', mode='rb') as in_file:
        return pickle.load(in_file)


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    """
    vocab = set(text.split())
    vocab_to_int = copy.copy(CODES)

    for v_i, v in enumerate(vocab, len(CODES)):
        vocab_to_int[v] = v_i

    int_to_vocab = {v_i: v for v, v_i in vocab_to_int.items()}

    return vocab_to_int, int_to_vocab


def save_params(params):
    """
    Save parameters to file
    """
    with open('params.p', 'wb') as out_file:
        pickle.dump(params, out_file)


def load_params():
    """
    Load parameters from file
    """
    with open('params.p', mode='rb') as in_file:
        return pickle.load(in_file)

def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    source_id_text = []
    target_id_text = []
    for text in source_text.split('\n'):
        source_id_text.append([source_vocab_to_int[w] for w in text.split()])
        
    for text in target_text.split('\n'):
        target_id_text.append([target_vocab_to_int[w] for w in text.split()] + [target_vocab_to_int['<EOS>']])

    return source_id_text, target_id_text


def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths



source_path = 'data/test_en_fr/small_vocab_en'
target_path = 'data/test_en_fr/small_vocab_fr'
source_text = load_data(source_path)
target_text = load_data(target_path)
preprocess_and_save_data(source_path, target_path, text_to_ids)
(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = load_preprocess()
save_path = 'checkpoints/dev'
(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = load_preprocess()
max_target_sentence_length = max([len(sentence) for sentence in source_int_text])

train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]
valid_source = source_int_text[:batch_size]
valid_target = target_int_text[:batch_size]
(valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths ) = next(get_batches(valid_source,
                                                                                                             valid_target,
                                                                                                             batch_size,
                                                                                                             source_vocab_to_int['<PAD>'],
                                                                                                             target_vocab_to_int['<PAD>']))                                                                                                  

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

                yield padding_batch_inputs, batch_inputs_lens, padding_batch_targets, batch_targets_lens
            
            if not forever:
                break

    
    def _parse_dict(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as fp:
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
        with open(file_path, 'r', encoding='utf-8') as fp:
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

        # # Create dictionary
        # self.encoder_int_to_vocab, self.encoder_vocab_to_int = self._parse_dict(encode_file_path)
        # self.decoder_int_to_vocab, self.decoder_vocab_to_int = self._parse_dict(decode_file_path)

        # # Create seqs
        # self.encode_seqs = self._parse_seq(encode_file_path, self.encoder_vocab_to_int)
        # self.decode_seqs = self._parse_seq(decode_file_path, self.decoder_vocab_to_int)

        if not hasattr(self, 'sess'):
            # Train model from start
            print('Train new model')

            inputs = tf.placeholder(tf.int32, [None, None], name='input')
            targets = tf.placeholder(tf.int32, [None, None])
            lr = tf.placeholder(tf.float32)
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            target_sequence_length = tf.placeholder(tf.int32, [None,], name='target_sequence_length')
            max_target_sequence_length = tf.reduce_max(target_sequence_length)
            source_sequence_length = tf.placeholder(tf.int32, [None, ], name='source_sequence_length')
            dec_input = tf.strided_slice(targets, (0, 0), (T_BATCH_SIZE, -1), (1, 1))
            dec_input = tf.concat([tf.fill([T_BATCH_SIZE, 1], target_vocab_to_int['<GO>']), dec_input], 1)


            #### ENCODER ####
            cell = tf.contrib.rnn.MultiRNNCell([self._rnn_cell(ENCODER_RNN_SIZE, DROPOUT_KEEP_PROB) for _ in range(ENCODER_RNN_LAYERS_N)])
            enc_input = tf.contrib.layers.embed_sequence(inputs, len(source_vocab_to_int), EMBEDDING_DIM)
            _, encoder_state = tf.nn.dynamic_rnn(cell, enc_input, sequence_length=source_sequence_length, dtype=tf.float32)


            #### DECODER ####
            dec_cell = tf.contrib.rnn.MultiRNNCell([self._rnn_cell(ENCODER_RNN_SIZE, DROPOUT_KEEP_PROB) for _ in range(ENCODER_RNN_LAYERS_N)])
    
            embed = tf.Variable(tf.random_uniform([len(target_vocab_to_int), EMBEDDING_DIM]))
            dec_embed_input = tf.nn.embedding_lookup(embed, dec_input)
    
            output_layer = tf.layers.Dense(len(target_vocab_to_int), kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            train_helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, target_sequence_length)
            decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, train_helper, encoder_state, output_layer)
            train_output = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=max_target_sequence_length)[0]

            start_token = tf.tile([tf.constant(target_vocab_to_int['<GO>'], dtype=tf.int32)], [I_BATCH_SIZE])
            infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embed, start_token, target_vocab_to_int['<EOS>'])
            decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, infer_helper, encoder_state, output_layer)
            infer_output = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=max_target_sequence_length)[0]


            training_logits = tf.identity(train_output.rnn_output, name='logits')
            inference_logits = tf.identity(infer_output.sample_id, name='predictions')

            masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

            with tf.name_scope("optimization"):
                # Loss function
                cost = tf.contrib.seq2seq.sequence_loss(
                    training_logits,
                    targets,
                    masks)

                # Optimizer
                optimizer = tf.train.AdamOptimizer(lr)

                # Gradient Clipping
                gradients = optimizer.compute_gradients(cost)
                capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
                train_op = optimizer.apply_gradients(capped_gradients)


            # encode_pad_id = self.encoder_vocab_to_int['<PAD>']
            # decode_pad_id = self.decoder_vocab_to_int['<PAD>']
            # batch_generator = self._padding_batch(self.encode_seqs, self.decode_seqs, T_BATCH_SIZE, encode_pad_id, decode_pad_id)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                for epoch_i in range(EPOCH):
                    for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                            get_batches(train_source, train_target, batch_size,
                                        source_vocab_to_int['<PAD>'],
                                        target_vocab_to_int['<PAD>'])):

                        print(source_batch)
                        print(sources_lengths)

                        _, loss = sess.run(
                            [train_op, cost],
                            {inputs: source_batch,
                             targets: target_batch,
                             lr: learning_rate,
                             target_sequence_length: targets_lengths,
                             source_sequence_length: sources_lengths,
                             keep_prob: keep_probability})

                        print(loss)

                
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

        with open(file_path, 'r', encoding='utf-8') as fp:
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
    encode_file_path = './data/test_en_fr/small_vocab_en'
    decode_file_path = './data/test_en_fr/small_vocab_fr'

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
