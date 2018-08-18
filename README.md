# Deep show off
## Lite Core
It's a lite model that can be trained on two relevant sequences. After training, the model can predict sequence given another sequence.

## Usage
```python
from liteSeq2Seq import Seq2seq
model = Seq2seq()
model.train(encode_file_path, decode_file_path)
prediction_str = model.predict(encode_str)
```

