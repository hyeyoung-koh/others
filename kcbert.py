from transformers import BertTokenizer, BertModel
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
import pandas as pd
import sklearn
import librosa
import torch
import random
import time
import librosa.display
from transformers import AutoTokenizer, AutoModelWithLMHead
import os
import pandas as pd
from pprint import pprint
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning import LightningModule, Trainer, seed_everything
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import emoji
from soynlp.normalizer import repeat_normalize

# GPU / CPU 할당 코드
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

mycsv=pd.read_csv('')
dataset=[]
labels=[]

dataset.append()
labels.append()

real_dataset=TensorDataset(torch.tensor(dataset),torch.tensor(labels))

# train vs validation = 9:1로 분리
train_size = int(0.9 * len(real_dataset))
val_size = len(real_dataset) - train_size

# dataset을 train_dataset과 val_dataset으로 분리
train_dataset, val_dataset = random_split(real_dataset, [train_size, val_size])

# dataloader 기반 batch 생성
train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=32,
)
print(len(train_dataloader))

validation_dataloader = DataLoader(
    val_dataset,
    sampler=SequentialSampler(val_dataset),
    batch_size=32,
)
print(len(validation_dataloader))

'''모델 로드 및 FT'''
#모델 불러오기
import torch.nn
#model_a = BertModel.from_pretrained('bert-base-multilingual-uncased') #bert
model_a= AutoModelWithLMHead.from_pretrained("beomi/kcbert-base") #kcbert
tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base") #kcbert

hidden_size = 6768 #??
model_b=torch.nn.Sequential(torch.nn.Dropout(0.1),torch.nn.Linear(hidden_size,8))
model_c=torch.nn.Sequential(model_a,model_b)
model_c.cuda()

# 모델의 파라미터를 가져오기
params = list(model_c.named_parameters())

# 계층별 param 수 출력
print('Our model_c has {:} different named para meters.\n'.format(len(params)))
print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')
for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Output Layer ====\n')
for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))





