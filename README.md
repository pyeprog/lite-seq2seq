# lite-seq2seq - a super light weight seq2seq core

This project is aimed at creating a super easy way to put a modern seq2seq model in use. We want to keep it simple, handy with sufficient ability to fitting complex sequence transformation process.
The default model equips with bucket, attention mechanism, stack bidirectional encoder and beam search. Experiment on some test translation dataset indicates that the fitting ability of our default model is much better than that of vanilla seq2seq.

## Prerequisite
- Only Tensorflow (>1.4)

## Features
- Single file module,which means only liteSeq2Seq.py is necessary. Super easy to bring its power to other projects.
- Only 1 dependency -- tensorflow. It's safe to be stable version(>1.4). Version earlier than that has not been tested.
- Super simple API. You can remember them in seconds, `train` and `predict`.
- Flexibility remains. You can tune the model as much as you want. Nearly all parameters are configurable.
- An CLI app inside, which means you can play with it as soon as you get tensorflow installed.
- Nice ability to fit complex sequence transformation such as translation, chat or create couplet if you please.

## Data Format
Unfortunately, our model can only parse data in special format. Not so special actually :)
For seq2seq model receive sequence A and learn to return sequence B, our model can ONLY use two files as traning input, with one file containing all sequence A's, one sequence at a line and the other containing all sequence B's, one sequence at a line.
### Data Format Example
for model to learn how to transform English sequence A -> French sequence B
FILE that contain all English sequence A's
```
new jersey is sometimes quiet during autumn , and it is snowy in april .
the united states is usually chilly during july , and it is usually freezing in november .
california is usually quiet during march , and it is usually hot in june .
the united states is sometimes mild during june , and it is cold in september .
```

FILE that contain all French sequence B's
```
new jersey est parfois calme pendant l' automne , et il est neigeux en avril .
les états-unis est généralement froid en juillet , et il gèle habituellement en novembre .
california est généralement calme en mars , et il est généralement chaud en juin .
les états-unis est parfois légère en juin , et il fait froid en septembre .
```

There is a text processor inside liteSeq2Seq module called TextProcessor, so you can leave to it those trivial treadmill, such as process those punctuations with some seperated from text and others removed.
The only thing you should keep in mind is to ensure lines in both files are correlated. You can check it out in example above, line 1 of A file relates to line 1 of B file, line 2 of A file relates to line 2 of B file, etc.
To make lines correlated also means number of lines of both file should be equal. If not, it will raise an error.

## Text Process
As mentioned earlier, there is a simple text processor called TextProcessor inside the module, which might offer some help processing the text.
### Basic usage
```python
# Basic usage: 
# read from file, process each line and save the processed text in the same 'file_path'
# The origin file will be renamed 'file_path.origin'
tp = TextProcessor()
tp.read(file_path).process(inplace=True)
```
As default the text process includes several subprocesses below:
1. Remove all (), [], {} and text within them.
2. Transfer all x.x. or x.x to xx. For instance, p.m. -> pm or A.M to AM
3. Transfer : - / \ * & $ # @ ^ and ... alike(more than 2 dots) to ' '
4. Seperate text and the punctuations such as `yes,` `good.` `amazing!` `pardon?` `cool;` to `yes , ` `good . ` `amazing ! ` `pardon ? ` `cool ; `
5. Remove singular = < > " ` ( ) [ ] { } that float in text
6. Remove those singular ' that float in text. Those ' form words, like Lily's, o'clock will remain.

Subprocesses will be executed in sequence. 

### Customize subprocess
In some case you want to add more subprocess method or build your own process methods stack, it's relatively simple to do so.
```python
# You own method for example
def proc1(string):
  return string.lower()

def proc2(string):
  return string.replace('_', '-')

tp = TextProcessor()
tp.append(proc1)
tp.append(proc2)

# Process the file1 with default methods + proc1 + proc2
tp.read(file1_path).process(inplace=True)

# Process file2 with only proc1 and proc2, 
tp.read(file2_path).process(proc_fn_list=[proc1, proc2], inplace=True)
```

### Other usages
```python
tp = TextProcessor()

# Overwrite the processed content into origin file, thus no file_path.origin will not be generated 
tp.read(file_path).process(inplace=True, overwrite=True)

# If you want to get the processed content instead of writing content into file, you can do
# content_lines will be a list of strings. Each string represents one line of the file.
content_lines = tp.read(file_path).process(inplace=False)

# Custom methods are available
new_content_lines = tp.read(file_path).process([proc1, proc2], inplace=False)

# Process only a single string
processed_string = tp.process_str(string_of_one_line)

# Custom methods are also available
processed_string = tp.process_str(string_of_one_line, proc_fn_list=[proc1, proc2])
```


## Use Seq2seq in your code
### Basic usage
```python
from liteSeq2Seq import Seq2seq
from liteSeq2Seq import TextProcessor

# Process the text if you need
tp = TextProcessor()
tp.read('input_str_file').process(inplace=True)
tp.read('target_str_file').process(inplace=True)

model = Seq2seq()
model.train('input_str_file', 'target_str_file')
encode_str = input()
prediction_str = model.predict(encode_str)
```

It's very simple to use Seq2seq, under default hyper-parameters, you just need to specify two file path to train method and use predict method to transform text. 
After training, you model will be saved automatically at `<liteSeq2Seq.py dir>/models/<model_id>`. 

If you want to save the model to other dir. Use `Seq2seq.set_model_dir('new_dir')` before creating instance of Seq2seq. 

Use `model.get_id()` to get your model's default id. The default id will be a unique sequence of 20 numbers

Use `model.set_id(new_id)` to set your own id. The default id will be overwrited.

### Load model for prediction
```python
model = Seq2seq()
model.load('./models/<pre_trained_model_id>')
encode_str = input()
prediction_str = model.predict(encode_str)
```

### Load existed model and continue training
```python
model = Seq2seq()
model.train('other_input_str_file', 'other_target_str_file', './models/<pre_trained_model_id>')
```

### Change the hyperparameters
```python
# Hyperparameters
model = Seq2seq(
    embedding_dim=512,
    rnn_layer_size=1024,
    n_rnn_layers=3, 
    beam_width=3, 
    keep_prob=0.8, 
    valid_portion=0.05, 
    train_batch_size=32, 
    infer_batch_size=1, 
    max_gradient_norm=5.0, 
    epoch=10,
    max_global_step=float('inf'),
    learning_rate=1e-3,
    decay_rate=0.5,
    decay_every=1e3,
    decay_start_at=8e3,
    n_buckets=50,
    vocab_remain_rate=0.97,
    report_every=50,
    show_every=200,
    summary_every=50,
    save_every=500
    )
```
| Hyperparameter    | Type      | Description                                                  |
| ----------------- | --------- | ------------------------------------------------------------ |
| embedding_dim     | int       | Embedding layer size                                         |
| rnn_layer_size    | int       | Single lstm layer size, EVEN NUMBER ONLY, set for both encoder and decoder |
| n_rnn_layers      | int       | Number of layers of lstm network, set for both encoder and decoder |
| beam_width        | int       | Width of beam search                                         |
| keep_prob         | float     | Keep probability for each rnn node                           |
| valid_portion     | float     | Portion seperated for validtion                              |
| train_batch_size  | int       | Batch size while training                                    |
| infer_batch_size  | int       | Batch size while infering                                    |
| max_gradient_norm | float     | Clip value for global gradients                              |
| epoch             | int       | Number of training epoch                                     |
| max_global_step   | int/float | Maximum training steps, default to infinity, which means training for {epoch} times |
| learning_rate     | float     | The learning rate                                            |
| decay_rate        | float     | The decay rate of learning rate                              |
| decay_every       | int       | For every {this} steps, learning_rate=learning_rate * decay_rate |
| decay_start_at    | int       | The learning rate begin to decay after training {this} number of steps |
| n_buckets         | int       | Seperate training sequence into {this} buckets, training sequences in same bucket have similar length |
| vocab_remain_rate | float     | Choose a vocab size that can cover {this} percentage of total words |
| report_every      | int       | Print validation score for every {this} steps                |
| show_every        | int       | Print example of transformation for every {this} steps       |
| summary_every     | int       | Save summery info for tensorboard for every {this} steps     |
| save_every        | int       | Save checkpoint for every {this} steps                       |

## Use Seq2seq via CLI
In terminal you can enter `python liteSeq2Seq.py -h` or `python liteSeq2Seq.py --help` for more info. 
### Start training
If you specify --enc, --dec, you will start to train a new model
```terminal
python liteSeq2Seq.py --enc 'path of input_str_file' --dec 'path of output_str_file'
```

### Continue training
If you specify --enc, --dec, --model, you will continue training existed model. Enter absolute path for --model.
```terminal
python liteSeq2Seq.py --enc 'path of input_str_file' --dec 'path of output_str_file' --model './models/model_id'
```

### Prediction
Specify --model to load a existed model, Specify --input for one-line prediction; Specify --loop for continuous prediction.
```terminal
python liteSeq2Seq.py --model './models/model_id' --input 'hello there'
python liteSeq2Seq.py --model './models/model_id' --loop
```

### Customize the model for training
Specify hyperparameters with --hyperparameters. For example, you want to specify beam_width, just add `--beam_width 10`
```terminal
python liteSeq2Seq.py --enc 'path of input_str_file' --dec 'path of output_str_file' --beam_width 10 --save_every 20 --keep_prob 0.5
```

## Tensorboard
Tensorboard is available and information is gathered for every {summary_every} steps. Summary info of each model is saved beside its checkpoint file.
You can simply launch tensorboard sever in terminal. Specify --logdir with the path you save your models in.
```terminal
tensorboard --logdir .models/
```

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

> Authors: Weilong Liao, Yimo Wang
