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

# GPU / CPU 할당 코드
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# 음성 특징 추출 (MFCC)
def mfcc_feature(myfile):  # myfile: 한 문장에 대한 wav 파일
    # MFCC 추출
    y, sr = librosa.load(myfile, sr=None)
    mfcc_extracted = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    print('myfile명:', myfile)
    mfcc_extracted = torch.from_numpy(mfcc_extracted).float()
    print('추출한 mfcc:', mfcc_extracted)
    print('mfcc_extracted의 shape:', mfcc_extracted.shape)
    print('mfcc_extracted.size(0):', mfcc_extracted.size(0))

    # scaling & padding
    mfcc_extracted_reshape = mfcc_extracted.view(1, -1)
    print('mfcc_extracted_reshape:',mfcc_extracted_reshape)
    print('mfcc_extracted_reshape의 shape:',mfcc_extracted_reshape.shape)

    mfcc_scale = sklearn.preprocessing.scale(mfcc_extracted_reshape, axis=1)
    print('mfcc_scale:', mfcc_scale)
    print('mfcc_scale의 shape:', mfcc_scale.shape)

    pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))
    padded_mfcc = pad2d(mfcc_scale, 6000)
    print('padded_mfcc:', padded_mfcc)
    print('padded_mfcc의 shape:', padded_mfcc.shape)
    print('mfcc_extracted_reshape의 shape:', mfcc_extracted_reshape.shape)

    padded_mfcc = torch.Tensor(padded_mfcc)
    print(padded_mfcc)

    return padded_mfcc

mfcc_feature('C:\\Users\\hyeyoung\\PycharmProjects\\test\\melspectrogram\\clip1_0_cut.wav')

#시각화
# file_display='C:\\Users\\hyeyoung\\PycharmProjects\\test\\melspectrogram\\clip1_0_cut.wav'
# y, sr = librosa.load(file_display, sr=None)
# mfcc_extracted = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
# fig, ax = plt.subplots()
# img = librosa.display.specshow(mfcc_extracted, x_axis='time', ax=ax)
# fig.colorbar(img, ax=ax)
# ax.set(title='MFCC')

# 텍스트 특징 추출 (BERT)
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model_bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
def text_feature(aihub_inputs,tokenizer=tokenizer_bert,model=model_bert):
    aihub_final_inputs = tokenizer(aihub_inputs, return_tensors='pt')
    outputs = model(**aihub_final_inputs)
    last_hidden_states = outputs.last_hidden_state
    cls_token = last_hidden_states[0][0]
    print('cls_token:',cls_token)
    print('cls_token shape:',cls_token.shape)
    cls_token_final=cls_token.reshape(1,-1)
    print('cls_token_final의 shape:', cls_token_final.shape)

    return cls_token_final

mycsv=pd.read_csv('C:\\Users\\hyeyoung\\PycharmProjects\\test\\melspectrogram\\clip_1myfinal.csv',encoding='utf-8-sig')
text_feature(mycsv.loc[1,'text_script'])
print('추출한 문장:',mycsv.loc[0,'text_script'])

dataset = []  # concat vector들을 삽입하는 리스트
labels = []  # 대응되는 label들을 삽입하는 리스트

# clip10개에 대해 한 문장씩 불러오기
for i in range(1, 10):  # i : clip num
    mycsv = pd.read_csv('C:\\Users\\hyeyoung\\PycharmProjects\\test\\melspectrogram\\clip_' + str(i) + 'myfinal.csv',encoding='utf-8-sig')
    print(i)

    for j in range(0, len(mycsv)):  # j : 행 idx
        print('j:', j)
        # 한 문장에 대한 padded mfcc 추출
        myfile = 'C:\\Users\\hyeyoung\\PycharmProjects\\test\\melspectrogram\\clip' + str(i) + '_' + str(j) + '_cut.wav'
        mfcc_padded_feature = mfcc_feature(myfile)
        print(type(mfcc_padded_feature))  # torch.tensor이다.

        # 한 문장에 대한 cls hs 추출
        text_feature_extract = text_feature(mycsv.loc[j, 'text_script'])

        # 각 vector을 np로 변환
        numpy_mfcc_feature = mfcc_padded_feature.numpy()
        print(numpy_mfcc_feature)
        numpy_text_feature = text_feature_extract.detach().numpy()  # @@
        print(numpy_text_feature)

        # numpy_mfcc_feature와 numpy_text_feature를 concat
        concat = np.concatenate((numpy_mfcc_feature, numpy_text_feature), axis=1)
        dataset.append(concat)

        # labels값 불러오기 (labels: 감정을 정수로 표현한 값들의 리스트 0-7)
        labels.append(torch.tensor(mycsv.loc[j, 'emotion_num']))

print('labels:', labels)
print(len(labels))
print(dataset)
print(len(dataset))

torch.tensor(dataset).size()
torch.tensor(labels).size()
real_dataset=TensorDataset(torch.tensor(dataset),torch.tensor(labels))

# train vs validation = 9:1로 분리
train_size = int(0.9 * len(real_dataset))
val_size = len(real_dataset) - train_size

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

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
model_a = BertModel.from_pretrained('bert-base-multilingual-uncased')
hidden_size = 6768
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


'''Fine-Tuning를 위한 환경 설정'''
optimizer = AdamW(model_c.parameters(), lr=2e-5, eps=1e-8)
epochs = 6
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

'''accuracy 반환하는 함수'''
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    print('preds:',preds)
    print('pred_flat:', pred_flat)
    print('np.argmax(preds,axis=2):',np.argmax(preds,axis=2))
    print('labels:',labels)
    labels_flat = labels.flatten()
    print('labels_flat:',labels_flat)
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


'''모델 학습'''
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

training_stats = []

num_labels = 8
hidden_size = 6768
dropout = torch.nn.Dropout()
classifier = torch.nn.Linear(hidden_size, 8)  # num_labels=8

from torch.nn import CrossEntropyLoss
loss_fct=CrossEntropyLoss()
print(dataset)

for epoch_i in range(0, epochs):
    # ========================================
    #               Training(학습)
    # ========================================
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    total_train_loss = 0
    num_labels = 8
    hidden_size = 6768
    model_c.train()
    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            print('batch{:>5,} of {:>5,}.'.format(step, len(train_dataloader)))
        b_labels = batch[1].to(device)
        print(b_labels)
        print(len(b_labels))
        model_c.zero_grad()
        pooled_output=batch[0]
        pooled_output = dropout(torch.tensor(pooled_output))
        logits = classifier(pooled_output)
        print('logits:', logits)
        print(logits.shape)
        print(labels)
        logits2=logits.to(device)
        batch[1]=torch.tensor(batch[1])
        batch[1]=batch[1].to(device)
        loss=loss_fct(logits2.view(-1,num_labels),batch[1])
        print('loss:', loss)

        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_c.parameters(), 1.0)
        optimizer.step()

        #학습률 update
        scheduler.step()

    #average loss 계산
    avg_train_loss = total_train_loss / len(train_dataloader)


    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    # ========================================
    #               Validation
    # ========================================
    print("")
    print("Running Validation...")
    t0 = time.time()
    model_c.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    num_labels = 8
    hidden_size = 6768

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        b_labels=batch[1].to(device)
        with torch.no_grad():
            pooled_output = batch[0]  # batch_concat
            pooled_output = dropout(torch.tensor(pooled_output))
            logits = classifier(pooled_output)
            print('logits:', logits)
            print(logits.type)  # Tensor object
            print(logits.shape)
            logits2 = logits.to(device)
            batch[1] = torch.tensor(batch[1])
            batch[1] = batch[1].to(device)
            print(batch[1])
            loss = loss_fct(logits2.view(-1, num_labels), batch[1])
            print('loss:', loss)

        # Accumulate the validation loss.
        total_eval_loss += loss.item()
        # Move logits and labels to CPU
        logits3 = logits2.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits3, label_ids)
        print(flat_accuracy(logits3,label_ids))
        print(total_eval_accuracy)

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
        }
    )
print('epoch:',training_stats[0]['epoch'])
print('Training Loss:',training_stats[0]['Training Loss'])
print('Validation Loss:',training_stats[0]['Valid. Loss'])
print('Validation Accuracy:',training_stats[0]['Valid. Accur.'])

print('epoch:',training_stats[1]['epoch'])
print('Training Loss:',training_stats[1]['Training Loss'])
print('Validation Loss:',training_stats[1]['Valid. Loss'])
print('Validation Accuracy:',training_stats[1]['Valid. Accur.'])

print('epoch:',training_stats[2]['epoch'])
print('Training Loss:',training_stats[2]['Training Loss'])
print('Validation Loss:',training_stats[2]['Valid. Loss'])
print('Validation Accuracy:',training_stats[2]['Valid. Accur.'])

print('epoch:',training_stats[3]['epoch'])
print('Training Loss:',training_stats[3]['Training Loss'])
print('Validation Loss:',training_stats[3]['Valid. Loss'])
print('Validation Accuracy:',training_stats[3]['Valid. Accur.'])

print('epoch:',training_stats[4]['epoch'])
print('Training Loss:',training_stats[4]['Training Loss'])
print('Validation Loss:',training_stats[4]['Valid. Loss'])
print('Validation Accuracy:',training_stats[4]['Valid. Accur.'])

# print('epoch:',training_stats[5]['epoch'])
# print('Training Loss:',training_stats[5]['Training Loss'])
# print('Validation Loss:',training_stats[5]['Valid. Loss'])
# print('Validation Accuracy:',training_stats[5]['Valid. Accur.'])

# print(training_stats[6]['epoch'])
# print(training_stats[6]['Training Loss'])
# print('Validation Loss:',training_stats[6]['Valid. Loss'])
# print('Validation Accuracy:',training_stats[6]['Valid. Accur.'])
#
# print(training_stats[7]['epoch'])
# print('Training Loss:',training_stats[7]['Training Loss'])
# print('Validation Loss:',training_stats[7]['Valid. Loss'])
# print('Validation Accuracy:',training_stats[7]['Valid. Accur.'])
#
# print(training_stats[8]['epoch'])
# print('Training Loss:',training_stats[8]['Training Loss'])
# print('Validation Loss:',training_stats[8]['Valid. Loss'])
# print('Validation Accuracy:',training_stats[8]['Valid. Accur.'])

