# Deep show off
## Lite Core
It's a lite model that can be trained on two relevant sequences. After training, the model can predict sequence given another sequence.
This project is aiming at provide a way to manipulate the power of seq2seq in an extreme simple way. Simplicity is the other facet of reusability. 
Moreover, the core should be easy to read and modify, and with enough explain about the missing detail in official API.

The core has ensembled several technics, including:
- Stack bidirectional rnn for encoder
- LuongAttention mechanism for both training and inference decoder
- Beam search for inference decoder

Several features are on schedule
- Auto stopping without specifying epoch
- Several text content cleaning utils

## Usage
```python
from liteSeq2Seq import Seq2seq
model = Seq2seq()
# model.load('./models/<model_id>') # if you want to load a trained model
model.train(encode_file_path, decode_file_path)
prediction_str = model.predict(encode_str)
```

