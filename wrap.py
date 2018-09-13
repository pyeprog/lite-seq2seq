from liteSeq2Seq import Seq2seq
from liteSeq2Seq import TextProcessor

def cut_short(x):
    if len(x) > 35:
        return x
    else:
        return ''

tp1 = TextProcessor()
tp1.append(cut_short)

lines = tp1.read('./data/movie_dialogs/enc').process(inplace=True)

