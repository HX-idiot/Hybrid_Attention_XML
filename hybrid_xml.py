
# coding: utf-8

# In[1]:


import os
import argparse
import math
import numpy as np
import timeit
import data_helpers
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# In[3]:


def load_data(data_path,max_length,vocab_size,batch_size=64):
    X_trn, Y_trn, X_tst, Y_tst, vocabulary, vocabulary_inv = data_helpers.load_data(data_path, max_length=max_length, vocab_size=vocab_size)
    Y_trn = Y_trn[0:].toarray()
    Y_trn=np.insert(Y_trn,101,0,axis=1)
    Y_trn=np.insert(Y_trn,102,0,axis=1)
    Y_tst = Y_tst[0:].toarray()

    train_data = data_utils.TensorDataset(torch.from_numpy( X_trn).type(torch.LongTensor),torch.from_numpy( Y_trn).type(torch.LongTensor))
    test_data = data_utils.TensorDataset(torch.from_numpy(X_tst).type(torch.LongTensor),torch.from_numpy(Y_tst).type(torch.LongTensor))
    train_loader = data_utils.DataLoader(train_data, batch_size, drop_last=True)
    test_loader = data_utils.DataLoader(test_data, batch_size, drop_last=True)
    return train_loader, test_loader,vocabulary, X_tst, Y_tst, X_trn,Y_trn


# In[4]:


#input data_path
data_path='/data/rcv1_raw_text.p'
sequence_length=500
vocab_size=30000
batch_size=64


# In[5]:



print('-'*50)
print('Loading data...'); start_time = timeit.default_timer()
train_loader, test_loader,vocabulary, X_tst, Y_tst, X_trn,Y_trn= load_data(data_path,sequence_length,vocab_size,batch_size)
print('Process time %.3f (secs)\n' % (timeit.default_timer() - start_time))


# In[5]:


def load_glove_embeddings(path, word2idx, embedding_dim):
    """Loading the glove embeddings"""
    with open(path) as f:
        embeddings = np.zeros((len(word2idx), embedding_dim))
        for line in f.readlines():
            values = line.split()
            word = values[0]
            index = word2idx.get(word)
            if index:
                vector = np.array(values[1:], dtype='float32')
                if vector.shape[-1] != embedding_dim:
                    raise Exception('Dimension not matching.')
                embeddings[index] = vector
        return torch.from_numpy(embeddings).float()


# In[6]:


#load glove 
pretrain='glove'
embedding_dim=300
if pretrain=='glove':
    #input word2vec file path
    file_path=os.path.join('glove.6B','glove.6B.%dd.txt'%(embedding_dim))
    embedding_weights = load_glove_embeddings(file_path,vocabulary,embedding_dim)


# In[7]:


#create Network structure

class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))
        
    def load(self,path):
        self.load_state_dict(torch.load(path))
        
    def save(self,path=None):
        if path is None:
            raise ValueError('Please specify the saving road!!!')
        torch.save(self.state_dict(),path)
        return path


# In[9]:


def get_embedding_layer(embedding_weights):
    word_embeddings=nn.Embedding(embedding_weights.size(0),embedding_weights.size(1))
    word_embeddings.weight.data.copy_(embedding_weights)
    word_embeddings.weight.requires_grad=False #not train
    return word_embeddings
        
class Hybrid_XML(BasicModule):
    def __init__(self,num_labels=3714,vocab_size=30001,embedding_size=300,embedding_weights=None,
                max_seq=300,hidden_size=256,d_a=256,label_emb=None):
        super(Hybrid_XML,self).__init__()
        self.embedding_size=embedding_size
        self.num_labels=num_labels
        self.max_seq=max_seq
        self.hidden_size=hidden_size
        
        if embedding_weights is None:
            self.word_embeddings=nn.Embedding(vocab_size,embedding_size)
        else:
            self.word_embeddings=get_embedding_layer(embedding_weights)

        self.lstm=nn.LSTM(input_size=self.embedding_size,hidden_size=self.hidden_size,num_layers=1,batch_first=True,bidirectional=True)
        
        #interaction-attention layer
        self.key_layer = torch.nn.Linear(2*self.hidden_size,self.hidden_size)
        self.query_layer=torch.nn.Linear(self.hidden_size,self.hidden_size)
        
        #self-attn layer
        self.linear_first = torch.nn.Linear(2*self.hidden_size,d_a)
        self.linear_second = torch.nn.Linear(d_a,self.num_labels)

        #weight adaptive layer
        self.linear_weight=torch.nn.Linear(2*self.hidden_size,1)
        
        #shared for all attention component
        self.linear_final = torch.nn.Linear(2*self.hidden_size,self.hidden_size)
        self.output_layer=torch.nn.Linear(self.hidden_size,1)
        
        label_embedding=torch.FloatTensor(self.num_labels,self.hidden_size)
        if label_emb is None:
            nn.init.xavier_normal_(label_embedding)
        else:
            label_embedding.copy_(label_emb)
        self.label_embedding=nn.Parameter(label_embedding,requires_grad=False)

    def init_hidden(self,batch_size):
        if torch.cuda.is_available():
            return (torch.zeros(2,batch_size,self.hidden_size).cuda(),torch.zeros(2,batch_size,self.hidden_size).cuda())
        else:
            return (torch.zeros(2,batch_size,self.hidden_size),torch.zeros(2,batch_size,self.hidden_size))
                
    def forward(self,x):
       
        emb=self.word_embeddings(x)
        
        hidden_state=self.init_hidden(emb.size(0))
        output,hidden_state=self.lstm(emb,hidden_state)#[batch,seq,2*hidden]
        

        #get attn_key
        attn_key=self.key_layer(output) #[batch,seq,hidden]
        attn_key=attn_key.transpose(1,2)#[batch,hidden,seq]
        #get attn_query
        label_emb=self.label_embedding.expand((attn_key.size(0),self.label_embedding.size(0),self.label_embedding.size(1)))#[batch,L,label_emb]
        label_emb=self.query_layer(label_emb)#[batch,L,label_emb]
        
        #attention
        similarity=torch.bmm(label_emb,attn_key)#[batch,L,seq]
        similarity=F.softmax(similarity,dim=2)
        
        out1=torch.bmm(similarity,output)#[batch,L,label_emb]
    
        #self-attn output
        self_attn=torch.tanh(self.linear_first(output)) #[batch,seq,d_a]
        self_attn=self.linear_second(self_attn) #[batch,seq,L]
        self_attn=F.softmax(self_attn,dim=1)
        self_attn=self_attn.transpose(1,2)#[batch,L,seq]
        out2=torch.bmm(self_attn,output)#[batch,L,hidden]


        factor1=torch.sigmoid(self.linear_weight(out1))
        
        factor2=1-factor1
        
        out=factor1*out1+factor2*out2
        
        out=F.relu(self.linear_final(out))
        out=torch.sigmoid(self.output_layer(out).squeeze(-1))#[batch,L]

        return out


# In[10]:


label_emb=np.zeros((103,256))
with open('./label_embedding/rcv.emb','r') as f:
    for index,i in enumerate(f.readlines()):
        if index==0:
            continue
        i=i.rstrip('\n')
        n=i.split(' ')[0]
        content=i.split(' ')[1:]
        label_emb[int(n)]=[float(value) for value in content]
        


    
# In[11]:

label_emb[-2:]=np.random.randn(2,256)


# In[12]:


label_emb=torch.from_numpy(label_emb).float()


# In[13]:


use_cuda=torch.cuda.is_available()
# use_cuda=False
torch.cuda.set_device(0)
torch.manual_seed(1234)
if use_cuda:
    torch.cuda.manual_seed(1234)


# In[14]:


model=hybrid_xml(num_labels=103,vocab_size=30001,embedding_size=300,embedding_weights=embedding_weights,
                max_seq=500,hidden_size=256,d_a=256,label_emb=label_emb)
# model.load('./rcv_log/rcv_9.pth')
if use_cuda:
    model.cuda()


# In[15]:


# for p in model.parameters():
#     print(p.requires_grad)
# \lambda p: p.requires_grad,model.parameters())


# In[16]:


params = list(model.parameters())
k = 0
for i in params:
    l = 1
    print("structure of current layer：" + str(list(i.size())))
    for j in i.size():
        l *= j
    print("sum of parameters：" + str(l))
    k = k + l
print("total sum of parameters：" + str(k))


def precision_k(pred, label, k=[1, 3, 5]):
    batch_size = pred.shape[0]
    
    precision = []
    for _k in k:
        p = 0
        for i in range(batch_size):
            p += label[i, pred[i, :_k]].mean()
        precision.append(p*100/batch_size)
    
    return precision

def ndcg_k(pred, label, k=[1, 3, 5]):
    batch_size = pred.shape[0]
    
    ndcg = []
    for _k in k:
        score = 0
        rank = np.log2(np.arange(2, 2 + _k))
        for i in range(batch_size):
            l = label[i, pred[i, :_k]]
            n = l.sum()
            if(n == 0):
                continue
            
            dcg = (l/rank).sum()
            label_count = label[i].sum()
            norm = 1 / np.log2(np.arange(2, 2 + np.min((_k, label_count))))
            norm = norm.sum()
            score += dcg/norm
            
        ndcg.append(score*100/batch_size)
    
    return ndcg


optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=0.001,weight_decay=4e-5)
criterion=torch.nn.BCELoss(reduction='sum')
epoch=15
best_acc=0.0
pre_acc=0.0


# if not os.path.isdir('./rcv_log'):
#     os.makedirs('./rcv_log')
# trace_file='./rcv_log/trace_rcv.txt'

for ep in range(1,epoch+1): 
    train_loss=0
    print("----epoch: %2d---- "%ep)
    model.train()
    for i,(data,labels) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        
        data=data.cuda()
        labels=labels.cuda()
        
        pred=model(data,label_emb)
        loss=criterion(pred,labels.float())/pred.size(0)
        loss.backward()
        optimizer.step()

        train_loss+=float(loss)
    batch_num=i+1
    train_loss/=batch_num
    
    print("epoch %2d 训练结束 : avg_loss = %.4f"%(ep,train_loss))
    print("开始进行validation")
    test_loss=0
    test_p1, test_p3, test_p5 = 0, 0, 0
    test_ndcg1, test_ndcg3, test_ndcg5=0, 0, 0
    model.eval()
    for i,(data,labels) in enumerate(tqdm(test_loader)):
        
        data=data.cuda()
        labels=labels.cuda()
        pred=model(data,label_emb)
        loss=criterion(pred,labels.float())/pred.size(0)

        #计算metric
        labels_cpu=labels.data.cpu()
        pred_cpu=pred.data.cpu()

        _p1,_p3,_p5=precision_k(pred_cpu.topk(k=5)[1].numpy(), labels_cpu.numpy(), k=[1,3,5])
        test_p1+=_p1
        test_p3+=_p3
        test_p5+=_p5


        _ndcg1,_ndcg3,_ndcg5=ndcg_k(pred_cpu.topk(k=5)[1].numpy(), labels_cpu.numpy(), k=[1,3,5])
        test_ndcg1+=_ndcg1
        test_ndcg3+=_ndcg3
        test_ndcg5+=_ndcg5

        test_loss+=float(loss)
    batch_num=i+1
    test_loss/=batch_num

    test_p1/=batch_num
    test_p3/=batch_num
    test_p5/=batch_num

    test_ndcg1/=batch_num
    test_ndcg3/=batch_num
    test_ndcg5/=batch_num

    print("epoch %2d 测试结束 : avg_loss = %.4f"%(ep,test_loss))
    print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f "%(test_p1,test_p3,test_p5))
    print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f "%(test_ndcg1,test_ndcg3,test_ndcg5))


    if test_p3<pre_acc:
        for param_group in optimizer.param_groups:
            param_group['lr']=0.0001
    pre_acc=test_p3
