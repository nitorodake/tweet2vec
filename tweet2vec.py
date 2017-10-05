import numpy as np
import argparse
import pickle
from tqdm import tqdm

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt

from chainer import optimizers, serializers, cuda, initializers
import chainer
import chainer.functions as F
import chainer.links as L

    
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')

parser.add_argument('--epoch', '-e', default=20, type=int,
                    help='number of epochs to learn')

parser.add_argument('--embed', '-em', default=150, type=int,
                    help='number of word embedding')

parser.add_argument('--unit', '-u', default=500, type=int,
                    help='number of lstm hidden units')

parser.add_argument('--dropout', '-d', default=0.25, type=float,
                    help='dropout rate')

parser.add_argument('--batchsize', '-b', default=100, type=int,
                    help='learning minibatch size')
args = parser.parse_args()

# GPU setting
print('====================')
if args.gpu >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()
    cuda.check_cuda_available()
    xp = cuda.cupy
    print(' Use GPU     : {}'.format(args.gpu))
else:
    xp = np
    print(' Use CPU')

#parameter
n_hidden  = args.unit
n_embed   = args.embed
n_epoch   = args.epoch
bs        = args.batchsize
drop_rate = args.dropout

# print parameter
print('====================')
print(' Unit            : {}'.format(n_hidden))
print(' Embed size      : {}'.format(n_embed))
print(' Epoch           : {}'.format(n_epoch))
print(' Minibatch       : {}'.format(bs))
print(' Dropout rate    : {}'.format(drop_rate))

class Tweet2Vec(chainer.Chain):
    def __init__(self, n_vocab, n_hashtag, n_embed, n_hidden):
        super(Tweet2Vec, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_embed, initialW=initializers.Uniform(1. / n_embed))
            self.gru   = L.NStepBiGRU(1, n_embed, n_hidden, drop_rate)
            self.lf    = L.Linear(n_hidden, n_hidden, initial_bias=0)
            self.lb    = L.Linear(n_hidden, n_hidden, initial_bias=0)
            self.out   = L.Linear(n_hidden, n_hashtag)

    def __call__(self, x, y):
        hx = None #隠れ

        xs = []
        for chars in x:
            e = self.embed(chars) # tweet内に含まれる文字の集合(chars)を一括でベクトルへ埋込み
            xs.append(e)        
            
        # last_o : 最終timestepの出力ベクトル (GRUの階層数*2,batchsize,hidden_size)
        # all_o  : 各timestepの出力ベクトル [ [入力数(1文の単語数),hidden_size], ...]
        last_o, all_o = self.gru(hx=hx, xs=xs)

        if chainer.config.train:
            loss = None
            f_last = last_o[0]
            b_last = last_o[1]
            te = self.lf(f_last)+self.lb(b_last)
            
            loss = F.sigmoid_cross_entropy(self.out(te),y)
            return loss
        

#=====================
#       main
#=====================

# データ読み込み
n_vocab   = 500 # それぞれのtweetは20~80文字で構成されるとする
n_hashtag = 200 # それぞれのtweetが1~3つのハッシュタグを持つとする

input_data = [ np.random.randint(0,500,n).astype(np.int32) for n in np.random.randint(20,80,1000)] # 今回は乱数で生成
label_data = [ np.random.randint(0,200,n) for n in np.random.randint(1,3,1000)]   # 今回は乱数で生成
n_data = len(input_data)

# model 定義
model = Tweet2Vec(n_vocab, n_hashtag, n_embed, n_hidden)
optimizer = optimizers.NesterovAG() # Nesterov’s momentum L2正則化未実装
optimizer.setup(model)

if args.gpu >= 0:
    model.to_gpu()
    cuda.get_device_from_id(args.gpu).use()

loss_list = [] #損失保存

for epoch in tqdm(range(n_epoch)):
    sffindx = np.random.permutation(n_data)
    for i in range(0, n_data, bs):
        sf_list = sffindx[i:(i+bs) if (i+bs) < n_data else n_data]

        model.zerograds()

        x = [input_data[index] for index in sf_list]

        tmp_y = []
        for index in sf_list:
            label_vec = np.zeros(n_hashtag)
            for label in label_data[index]:
                label_vec[label] = 1
            tmp_y.append(label_vec)

        y = np.array(tmp_y,dtype=np.int32)
        loss = model(x, y)    
        loss.backward()
        optimizer.update()

        if i == 0:
            loss_list.append(loss.data)

            
#with chainer.using_config('train', False):
         
#outfile = './test.model'
#serializers.save_npz(outfile, model) #モデル保存

#損失変化の描画
plt.plot(loss_list)
plt.ylabel('error')
plt.xlabel('epoch')
#plt.show()
plt.savefig('./loss.png')
