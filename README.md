# LiteSeq2Seq - a light weight seq2seq core
## The Core
It's a lite model that can be trained on two relevant sequences. After training, the model can predict sequence given another sequence.
This project is aiming at provide a way to manipulate the power of seq2seq simply. Simplicity is the other facet of reusability. 
Moreover, the core is easy to read and modify, and with enough explain and comment about the missing detail in official API.

The core has ensembled several technics, including:
- Stack bidirectional rnn for encoder
- LuongAttention mechanism for both training and inference decoder
- Beam search for inference decoder

Several features are on schedule
- Auto stopping without specifying epoch
- Several text content cleaning utils (working on...)

## Usage
```python
# Train new model from start
from liteSeq2Seq import Seq2seq
model = Seq2seq()
model.train('input_str_file', 'target_str_file')
encode_str = input()
prediction_str = model.predict(encode_str)


# Use pre-trained model to predict
model = Seq2seq()
model.load('./models/<pre_trained_model_id>')
encode_str = input()
prediction_str = model.predict(encode_str)


# Train a pre-trained model on other dataset
model = Seq2seq()
model.load('./models/<pre_trained_model_id>')
model.train('other_input_str_file', 'other_target_str_file')
```

## Current state
We meet some problem on training seq2seq with GPU. It works okay with CPU though.
