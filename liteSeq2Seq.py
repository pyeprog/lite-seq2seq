import tensorflow as tf

class Seq2seq:
    def __init__(self):
        pass

    def train(self, encode_file_path, decode_file_path):
        pass

    def predict(self, encode_str):
        return 'test'


if __name__ == '__main__':
    model = Seq2seq()
    encode_file_path = './data/test'
    decode_file_path = './data/test'
    encode_str = 'example'
    model.train(encode_file_path, decode_file_path)
    prediction_str = model.predict(encode_str)
    print("> {}".format(prediction_str))
