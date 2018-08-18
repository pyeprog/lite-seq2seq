# Deep show off
## Lite Core
It's a lite model that can be trained on two relevant sequences. After training, the model can predict sequence given another sequence.
This project is aiming at provide a way to manipulate the power of seq2seq in an extreme simple way. Simplicity is the other facet of reusability. 
Moreover, the core should be easy to read and modify, and with enough explain about the missing detail in official API.

## Usage
```python
from liteSeq2Seq import Seq2seq
model = Seq2seq()
model.train(encode_file_path, decode_file_path)
prediction_str = model.predict(encode_str)
```

