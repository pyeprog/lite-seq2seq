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
INPUT:  which i &apos;m glad you liked , but they did not like that at all .
PRED:  tôi thích lòng bạn bạn thích , , nhưng họ không không toàn không thích nhưng <PAD>
EXPECT:  rất vui vì các bạn thích nó , nhưng họ thì hoàn toàn không . <EOS> <PAD>
*********
1048/3801 E:3/10 B:8650 - train loss: 2.0453741550445557	valid loss: 2.2149105072021484	valid bleu: 0.27421582436549286	lr: 0.0005
1098/3801 E:3/10 B:8700 - train loss: 1.8414764404296875	valid loss: 1.7279936075210571	valid bleu: 0.38183637121628766	lr: 0.0005
1148/3801 E:3/10 B:8750 - train loss: 1.6540954113006592	valid loss: 1.8261561393737793	valid bleu: 0.32094779986631233	lr: 0.0005
1198/3801 E:3/10 B:8800 - train loss: 2.039867639541626	valid loss: 2.1421258449554443	valid bleu: 0.26091351867658935	lr: 0.0005
*********
INPUT:  and then you ask , why economists ? <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>
PRED:  và đến bạn hỏi hỏi tại tại sao các các nhà nhà kinh tế ? ? <EOS>
EXPECT:  có thể bạn sẽ hỏi : tại sao lại là các nhà kinh tế học ? <EOS>
*********
1248/3801 E:3/10 B:8850 - train loss: 1.8587589263916016	valid loss: 2.0005407333374023	valid bleu: 0.3392455093290396	lr: 0.0005
1298/3801 E:3/10 B:8900 - train loss: 1.891492486000061	valid loss: 1.9412884712219238	valid bleu: 0.3106838948851244	lr: 0.0005
1348/3801 E:3/10 B:8950 - train loss: 2.0899720191955566	valid loss: 2.053297996520996	valid bleu: 0.28445687192641694	lr: 0.0005
1398/3801 E:3/10 B:9000 - train loss: 1.7055124044418335	valid loss: 2.312941551208496	valid bleu: 0.22785045770699983	lr: 0.0005
*********
INPUT:  so that , if one thing changes , everything else changes . <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>
PRED:  vậy điều , thay thay điều thay đổi , mọi thứ khác thay thay đổi khác <PAD>
EXPECT:  để mà chỉ cần một thứ thay đổi là mọi thứ khác cũng thay đổi <EOS> <PAD>
*********
1448/3801 E:3/10 B:9050 - train loss: 1.1380536556243896	valid loss: 2.1460866928100586	valid bleu: 0.31229518563789693	lr: 0.00025
1498/3801 E:3/10 B:9100 - train loss: 0.920090913772583	valid loss: 1.8488413095474243	valid bleu: 0.29356650833508835	lr: 0.00025
1548/3801 E:3/10 B:9150 - train loss: 1.1864454746246338	valid loss: 2.1042449474334717	valid bleu: 0.23671743609433776	lr: 0.00025
1598/3801 E:3/10 B:9200 - train loss: 0.964065432548523	valid loss: 2.234126567840576	valid bleu: 0.22037351345615266	lr: 0.00025
*********
INPUT:  but , you know , we have become obsessed with this linear narrative . <PAD> <PAD>
PRED:  nhưng bạn ta đã bị ám kể về câu này nghĩ này tính này <EOS> <EOS> <PAD>
EXPECT:  nhưng chúng ta đã quá lệ thuộc vào kiểu suy nghĩ tuyến tính này . <EOS> <PAD>
*********
1648/3801 E:3/10 B:9250 - train loss: 1.7979393005371094	valid loss: 2.690335988998413	valid bleu: 0.20174338847822432	lr: 0.00025
1698/3801 E:3/10 B:9300 - train loss: 1.6464687585830688	valid loss: 2.469257116317749	valid bleu: 0.23382361948814526	lr: 0.00025
1748/3801 E:3/10 B:9350 - train loss: 1.6552934646606445	valid loss: 2.720841646194458	valid bleu: 0.14626498494873927	lr: 0.00025
1798/3801 E:3/10 B:9400 - train loss: 1.2420995235443115	valid loss: 2.548173666000366	valid bleu: 0.2009625139063891	lr: 0.00025
*********
INPUT:  it &apos;s been a severe disease for thousands of years . <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>
PRED:  đó hàng ngàn năm qua bệnh là bệnh là bệnh căn bệnh nghiêm trọng <EOS> <EOS> <PAD>
EXPECT:  trong hàng ngàn năm , nó được coi là 1 căn bệnh trầm trọng . <EOS> <PAD>
*********
1848/3801 E:3/10 B:9450 - train loss: 2.1727941036224365	valid loss: 2.2172937393188477	valid bleu: 0.275836290085675	lr: 0.00025
1898/3801 E:3/10 B:9500 - train loss: 2.0006470680236816	valid loss: 2.0729641914367676	valid bleu: 0.27740728153593047	lr: 0.00025
1948/3801 E:3/10 B:9550 - train loss: 1.7884482145309448	valid loss: 2.485726833343506	valid bleu: 0.19560731856258387	lr: 0.00025
1998/3801 E:3/10 B:9600 - train loss: 1.7653093338012695	valid loss: 2.1534786224365234	valid bleu: 0.2800324652154898	lr: 0.00025
*********
INPUT:  you need the time to get to know the people that you &apos;re serving . <PAD> <PAD> <PAD> <PAD>
PRED:  bạn cần thời gian để tìm được những người mà bạn đang phục vụ cho <EOS> <PAD>
EXPECT:  bạn cần thời gian để hiểu về những người mà bạn đang phục vụ . <EOS> <PAD>
*********
```
