
# coding: utf-8

# In[1]:


import os
import argparse
import numpy as np
import timeit
import data_helpers
import torch
import torch.utils.data as data_utils


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
batch_size=32


# In[5]:



print('-'*50)
print('Loading data...'); start_time = timeit.default_timer()
train_loader, test_loader,vocabulary, X_tst, Y_tst, X_trn,Y_trn ,catgy= load_data(data_path,sequence_length,vocab_size,batch_size)
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
import math
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# In[8]:


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
        
class hybrid_xml(BasicModule):
    def __init__(self,num_labels=103,vocab_size=30001,embedding_size=300,embedding_weights=None,
                max_seq=500,hidden_size=256,d_a=256,batch_size=batch_size):
        super(deepwalk_xml,self).__init__()
        self.embedding_size=embedding_size
        self.num_labels=num_labels
        self.max_seq=max_seq
        self.hidden_size=hidden_size
        self.batch_size=batch_size
        if embedding_weights is None:
            self.word_embeddings=nn.Embedding(vocab_size,embedding_size)
        else:
            self.word_embeddings=get_embedding_layer(embedding_weights)
            
        self.embedding_dropout=nn.Dropout(p=0.25,inplace=True)
        self.lstm=nn.LSTM(input_size=self.embedding_size,hidden_size=self.hidden_size,num_layers=1,batch_first=True,bidirectional=True)
        
        #interaction-attn layer
        self.key_layer = torch.nn.Linear(2*self.hidden_size,self.hidden_size)
        self.query_layer=torch.nn.Linear(self.hidden_size,self.hidden_size)
        
        #self-attn layer
        self.linear_first = torch.nn.Linear(2*self.hidden_size,d_a)
        self.linear_second = torch.nn.Linear(d_a,self.num_labels)
        
        #weight adaptive layer
        self.linear_weight1=torch.nn.Linear(2*self.hidden_size,1)
        self.linear_weight2=torch.nn.Linear(2*self.hidden_size,1)

        #prediction layer
        self.linear_final = torch.nn.Linear(512,256)
        self.output_layer=torch.nn.Linear(256,1)

    def init_hidden(self):
        if torch.cuda.is_available():
            return (torch.zeros(2,self.batch_size,self.hidden_size).cuda(),torch.zeros(2,self.batch_size,self.hidden_size).cuda())
        else:
            return (torch.zeros(2,self.batch_size,self.hidden_size),torch.zeros(2,self.batch_size,self.hidden_size))
                
    def forward(self,x,label_emb):
        emb=self.word_embeddings(x)
        emb=self.embedding_dropout(emb)#[batch,seq_len,embeddim_dim]

        hidden_state=self.init_hidden()
        output,hidden_state=self.lstm(emb,hidden_state)#[batch,seq,2*hidden]
        
        #get attn_key
        attn_key=self.key_layer(output) #[batch,seq,hidden]
        attn_key=attn_key.transpose(1,2)#[batch,hidden,seq]
        #get attn_query
        label_emb=label_emb.expand((attn_key.size(0),label_emb.size(0),label_emb.size(1)))#[batch,L,label_emb]
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
        out2=torch.bmm(self_attn,output)#[batch,L,2*hidden]
        
        #normalize
        out1=F.normalize(out1,p=2,dim=-1)
        out2=F.normalize(out2,p=2,dim=-1)

        factor1=torch.sigmoid(self.linear_weight1(out1))
        factor2=torch.sigmoid(self.linear_weight2(out2))
        
        factor1=factor1/(factor1+factor2)
        factor2=1-factor1
        
        out=factor1*out1+factor2*out2
        
        out=F.relu(self.linear_final(out),inplace=True)
        out=torch.sigmoid(self.output_layer(out).squeeze(-1))#[batch,L]

#         return out
        return out,factor1


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
                max_seq=500,hidden_size=256,d_a=256,batch_size=batch_size)
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


# In[17]:

def precision_k(true_mat, score_mat, k):
    p = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    for k in range(k):
        score_mat = np.copy(backup)
        for i in range(rank_mat.shape[0]):
            score_mat[i][rank_mat[i, :-(k + 1)]] = 0
        score_mat = np.ceil(score_mat)
        #         kk = np.argwhere(score_mat>0)
        mat = np.multiply(score_mat, true_mat)
        #         print("mat",mat)
        num = np.sum(mat, axis=1)
        p[k] = np.mean(num / (k + 1))
    return np.around(p, decimals=4)

def get_factor(label_count,k):
    res=[]
    for i in range(len(label_count)):
        n=int(min(label_count[i],k))
        f=0.0
        for j in range(1,n+1):
            f+=1/np.log2(j+1)
        res.append(f)
    return np.array(res)
        

def Ndcg_k(true_mat,score_mat,k):
    res=np.zeros((k,1))
    rank_mat=np.argsort(score_mat)
    backup=np.copy(score_mat)
    label_count=np.sum(true_mat,axis=1)
    
    for m in range(k):
        y_mat=np.copy(true_mat)
        for i in range(rank_mat.shape[0]):
            y_mat[i][rank_mat[i, :-(m + 1)]] = 0
            for j in range(m+1):
                y_mat[i][rank_mat[i,-(j+1)]] /= np.log2(j+1+1)
        
        dcg=np.sum(y_mat,axis=1)
        factor=get_factor(label_count,m+1)
        ndcg=np.mean(dcg/factor)
        res[m]=ndcg
    return np.around(res, decimals=4)


# In[18]:


optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=0.001,weight_decay=1e-5)
criterion=torch.nn.BCELoss(size_average=False)
epoch=15

# if not os.path.isdir('./rcv_log'):
#     os.makedirs('./rcv_log')
# trace_file='./rcv_log/trace_rcv.txt'

for ep in range(1,epoch+1):
    train_loss=[]
    prec_k=[]
    ndcg_k=[]
    print("----epoch: %2d---- "%ep)
    model.train()
    optimizer.zero_grad()
    for i,(data,labels) in enumerate(tqdm(train_loader)):
#         optimizer.zero_grad()
        if use_cuda:
            data=data.cuda()
            labels=labels.cuda()
            label_emb=label_emb.cuda()
        
#         pred,reg=model(data,label_emb)
        pred,factor=model(data,label_emb)
        loss_pred=criterion(pred,labels.float())/batch_size

        loss_pred.backward()
#         optimizer.step()
        if (i+1)%4==0:
            optimizer.step()
            optimizer.zero_grad()
        #calculate prec@1~5
        labels_cpu=labels.data.cpu().float()
        pred_cpu=pred.data.cpu()
        prec=precision_k(labels_cpu.numpy(),pred_cpu.numpy(),5)
        prec_k.append(prec)
        
        ndcg=Ndcg_k(labels_cpu.numpy(),pred_cpu.numpy(),5)
        ndcg_k.append(ndcg)

        train_loss.append(float(loss_pred))
    avg_loss=np.mean(train_loss)
    epoch_prec=np.array(prec_k).mean(axis=0)
    epoch_ndcg=np.array(ndcg_k).mean(axis=0)
    
    print("epoch %2d training end : avg_loss = %.4f"%(ep,avg_loss))
    print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f "%(epoch_prec[0],epoch_prec[2],epoch_prec[4]))
    print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f "%(epoch_ndcg[0],epoch_ndcg[2],epoch_ndcg[4]))
#     with open(trace_file,'a') as f:
#         f.write('epoch:{:2d} training end：loss:{:.4f} , p@1:{:.4f} , p@3:{:.4f}, p@5:{:.4f},     ndcg@1:{:.4f}, ndcg@3:{:.4f}, ndcg@5:{:.4f}'.format(ep,avg_loss,float(epoch_prec[0]),float(epoch_prec[2]),float(epoch_prec[4]),float(epoch_ndcg[0]),float(epoch_ndcg[2]),float(epoch_ndcg[4])))
#         f.write('\n')
    # if ep>6:
    print("begin validation")
    test_loss=[]
    test_acc_k=[]
    test_ndcg_k=[]
    model.eval()
    for (data,labels) in tqdm(test_loader):
        if use_cuda:
            data=data.cuda()
            labels=labels.cuda()
            label_emb=label_emb.cuda()

        pred,factor=model(data,label_emb)
        loss_pred=criterion(pred,labels.float())/batch_size

        #calculate prec@1~5
        labels_cpu=labels.data.cpu().float()
        pred_cpu=pred.data.cpu()
        prec=precision_k(labels_cpu.numpy(),pred_cpu.numpy(),5)
        test_acc_k.append(prec)

        ndcg=Ndcg_k(labels_cpu.numpy(),pred_cpu.numpy(),5)
        test_ndcg_k.append(ndcg)

        test_loss.append(float(loss_pred))
    avg_test_loss=np.mean(test_loss)
    test_prec=np.array(test_acc_k).mean(axis=0)
    test_ndcg=np.array(test_ndcg_k).mean(axis=0)
    print("epoch %2d testing end : avg_loss = %.4f"%(ep,avg_test_loss))
    print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f "%(test_prec[0],test_prec[2],test_prec[4]))
    print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f "%(test_ndcg[0],test_ndcg[2],test_ndcg[4]))
#     with open(trace_file,'a') as f:
#         f.write('epoch:{:2d} testing end：loss:{:.4f} , p@1:{:.4f} , p@3:{:.4f}, p@5:{:.4f},     ndcg@1:{:.4f}, ndcg@3:{:.4f}, ndcg@5:{:.4f}'.format(ep,avg_test_loss,float(test_prec[0]),float(test_prec[2]),float(test_prec[4]),float(test_ndcg[0]),float(test_ndcg[2]),float(test_ndcg[4])))
#         f.write('\n')
    # p='./rcv_log/rcv_%d.pth'%ep
    # name=model.save(path=p)
    # print("model save successfully!",name)
