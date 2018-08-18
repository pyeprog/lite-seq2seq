import tensorflow as tf
import os
from random import random

MODEL_PATH = './models'
if not os.path.isdir(MODEL_PATH):
    os.mkdir(MODEL_PATH)

class Seq2seq:
    def __init__(self):
        self._id = (str(random())[2:] + str(random())[2:])[:20]
        self.model_ckpt_dir = os.path.join(MODEL_PATH, self._id)
        self.model_ckpt_path = os.path.join(self.model_ckpt_dir, 'checkpoint.ckpt')

        if not os.path.isdir(self.model_ckpt_dir):
            os.mkdir(self.model_ckpt_dir)


    def train(self, encode_file_path, decode_file_path):
        pass


    def train_new(self, encode_file_path, decode_file_path):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Get placeholder
            # Get training_logits
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
