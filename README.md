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
- Several text content cleaning utils 
- Auto stopping without specifying epoch (working on...)

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


# If you have corpus, you can use TextProcessor to clean it.
# TextProcessor object has some default processing method.
from liteSeq2Seq import TextProcessor
tp = TextProcessor()
tp.read(file_path).process(inplace=True) #Save the processed data to file named `file_path.proc`
```

## Current state
It can be trained on CPU or GPU. 
Please keep in mind that it's pretty easy to use up your free memory if you have a large network or set the BATCH\_SIZE to a large number.

## Evaluation
### The translation result of English to Vietnamese
```
 *********
846/3801 E:5/10 B:16050 - train loss: 1.9108352661132812	valid loss: 1.8648494482040405	valid bleu: 0.3059511154884005	lr: 1.953125e-06
896/3801 E:5/10 B:16100 - train loss: 1.8717025518417358	valid loss: 1.826714038848877	valid bleu: 0.3140566889522327	lr: 1.953125e-06
946/3801 E:5/10 B:16150 - train loss: 1.8364272117614746	valid loss: 1.7648216485977173	valid bleu: 0.29352795471834864	lr: 1.953125e-06
996/3801 E:5/10 B:16200 - train loss: 1.5862809419631958	valid loss: 1.814572811126709	valid bleu: 0.33907792613568394	lr: 1.953125e-06
*********
INPUT:  <UNK> billion years pass and this song is still ringing all around us . <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>
PRED:  <UNK> mươi năm qua và bài hát này vẫn đang <UNK> <UNK> xung chúng ta . <PAD>
EXPECT:  14 tỉ năm qua và bài hát này vẫn luôn vang lên quanh chúng ta <EOS> <PAD>
*********
1046/3801 E:5/10 B:16250 - train loss: 1.3182300329208374	valid loss: 1.6934211254119873	valid bleu: 0.37053918847078254	lr: 1.953125e-06
1096/3801 E:5/10 B:16300 - train loss: 1.6285063028335571	valid loss: 1.8579593896865845	valid bleu: 0.35022817363585124	lr: 1.953125e-06
1146/3801 E:5/10 B:16350 - train loss: 1.2561579942703247	valid loss: 2.055617094039917	valid bleu: 0.2913682102088838	lr: 1.953125e-06
1196/3801 E:5/10 B:16400 - train loss: 1.4308795928955078	valid loss: 1.9623420238494873	valid bleu: 0.3123804461163495	lr: 1.953125e-06
*********
INPUT:  the thing to understand about shame is it &apos;s not guilt . <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>
PRED:  điều biết sự xấu hổ là nghĩa là nó không phải là lỗi lỗi . <EOS> <PAD>
EXPECT:  hiểu về sự xấu hổ có nghĩa là nó không phải là tội lỗi . <EOS> <PAD>
*********
1246/3801 E:5/10 B:16450 - train loss: 1.7905057668685913	valid loss: 1.9081865549087524	valid bleu: 0.3494880628701128	lr: 1.953125e-06
1296/3801 E:5/10 B:16500 - train loss: 1.589463233947754	valid loss: 1.6822949647903442	valid bleu: 0.3286996861980662	lr: 1.953125e-06
1346/3801 E:5/10 B:16550 - train loss: 1.6918792724609375	valid loss: 1.9649474620819092	valid bleu: 0.344897747790911	lr: 1.953125e-06
1396/3801 E:5/10 B:16600 - train loss: 1.4967548847198486	valid loss: 1.6828497648239136	valid bleu: 0.3509032703562231	lr: 1.953125e-06
*********
INPUT:  you want a cigarette in prison ? three to five dollars . <PAD> <PAD> <PAD> <PAD>
PRED:  bạn muốn thuốc loại thuốc ở tù tù ? ? <EOS> đến năm đô . . <EOS>
EXPECT:  bạn muốn một <UNK> thuốc ở trong tù ư ? 3 đến 5 đô la . <EOS>
*********
1446/3801 E:5/10 B:16650 - train loss: 0.8905309438705444	valid loss: 1.7163500785827637	valid bleu: 0.3482968483409197	lr: 1.953125e-06
1496/3801 E:5/10 B:16700 - train loss: 0.6797949075698853	valid loss: 1.7330551147460938	valid bleu: 0.3151641739022078	lr: 1.953125e-06
1546/3801 E:5/10 B:16750 - train loss: 0.8636561036109924	valid loss: 1.8132535219192505	valid bleu: 0.3593988877271352	lr: 1.953125e-06
1596/3801 E:5/10 B:16800 - train loss: 1.039635419845581	valid loss: 1.890971064567566	valid bleu: 0.348145940855399	lr: 1.953125e-06
*********
INPUT:  we take them out of the group , put them in a test chamber . <PAD> <PAD> <PAD> <PAD>
PRED:  chúng tôi đưa chúng ra khỏi nhóm , đặt chúng trong phòng kiểm nghiệm . <EOS> <PAD>
EXPECT:  chúng tôi tách chúng ra khỏi nhóm và mang chúng đến phòng thử nghiệm . <EOS> <PAD>
*********
1646/3801 E:5/10 B:16850 - train loss: 1.5933128595352173	valid loss: 1.9628973007202148	valid bleu: 0.3285307809158357	lr: 1.953125e-06
1696/3801 E:5/10 B:16900 - train loss: 1.1691449880599976	valid loss: 1.9147114753723145	valid bleu: 0.33175300873705554	lr: 1.953125e-06
1746/3801 E:5/10 B:16950 - train loss: 1.1576321125030518	valid loss: 1.822666883468628	valid bleu: 0.3125149585230503	lr: 1.953125e-06
1796/3801 E:5/10 B:17000 - train loss: 1.1822071075439453	valid loss: 2.1643805503845215	valid bleu: 0.30734826525996944	lr: 1.953125e-06
*********
INPUT:  it &apos;s <UNK> by being citizens , by being citizens , by being <UNK> . <PAD> <PAD> <PAD>
PRED:  nó là cách công án của công công dân , bởi là những dân , . <EOS>
EXPECT:  mà bằng chính vai trò là những công dân , đó là công dân ted . <EOS>
*********
```
