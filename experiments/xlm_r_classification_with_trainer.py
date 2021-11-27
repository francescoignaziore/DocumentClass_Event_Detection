from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification, BertTokenizer, BertTokenizerFast
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertTokenizerFast
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import __version__ as transformers_ver
from datasets import load_dataset
# import torch
import numpy as np
from pprint import pprint
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

#print("PyTorch Version: ", torch.__version__)
#print("PyTorch Cuda Version: ", torch.version.cuda)
#print("Tranformers Version: ", transformers_ver)
#print(torch.cuda.get_device_name(0))


##############
# Select Model
##############

# Try XLM-R
# select_model = 'xlm-roberta-base'   # xlm-roberta-large
# tokenizer = XLMRobertaTokenizer.from_pretrained(select_model)  # ISSUE: https://github.com/huggingface/transformers/issues/11095
# model = XLMRobertaForSequenceClassification.from_pretrained(select_model)
# model.config

# Try BERT
# select_model = 'bert-base-uncased'
# model = BertForSequenceClassification.from_pretrained(select_model)
# tokenizer = BertTokenizerFast.from_pretrained(select_model)
# model.config

# Try DistillBERT
select_model = 'distilbert-base-uncased'
model = DistilBertForSequenceClassification.from_pretrained(select_model, num_labels=2, force_download=False)
# tokenizer = DistilBertTokenizerFast.from_pretrained(select_model, force_download=False)
tokenizer = AutoTokenizer.from_pretrained(select_model, use_fast=True)
#model.config

print(f'                   Model: {model.config.model_type}')
print(f'         Vocabulary size: {model.config.vocab_size}')
print(f'Max input sequnce lenght: {model.config.max_position_embeddings}')


##############
# Load Dataset
##############
# Try shared task dataset
path_to_load_1 = '../../03_dataset/task_01/subtask1-document/en-train.json'
path_to_load_2 = '../../03_dataset/task_01/subtask1-document/es-train.json'
path_to_load_3 = '../../03_dataset/task_01/subtask1-document/pr-train.json'
train_dataset, val_dataset = load_dataset('json', data_files=[path_to_load_1,path_to_load_2, path_to_load_3] , split=['train[:80%]','train[80%:]'])
#train_dataset, val_dataset = load_dataset('json', data_files=[path_to_load_2] , split=['train[:80%]','train[80%:]'])

# Load IMDB dataset
# train_dataset, val_dataset = load_dataset('imdb', split=['train[:20%]', 'test[:20%]'])

# Load CoLA dataset
# train_dataset, test_dataset, val_dataset = load_dataset('glue','cola', split=['train','test','validation'])

# Rename columns to have 'sentence' column name
train_dataset = train_dataset.rename_column('text','sentence')
val_dataset = val_dataset.rename_column('text','sentence')

train_dataset.features
# print(train_dataset.features)
# print(len(train_dataset))
# pprint(train_dataset.info.__dict__)

print()
#print('Max sentence length: ', max([len(sen) for sen in train_dataset['sentence'] + test_dataset['sentence']]))
print('Max sentence length: ', max([len(sen) for sen in train_dataset['sentence'] + val_dataset['sentence']]))

# Set sequence lenght
sentence_lenght = 80

# Check embeddings
batch_sentences = train_dataset['sentence'][:3]
print(batch_sentences)
encoded = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt', max_length=sentence_lenght)
print(encoded)
print()
for ids in encoded["input_ids"]:
    pprint(tokenizer.decode(ids))


# Build embeddings
def tokenize(batch):
    return tokenizer(batch['sentence'], padding=True, truncation=True, max_length=sentence_lenght)  # batch['text']


train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
#test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))
val_dataset = val_dataset.map(tokenize, batched=True, batch_size=len(val_dataset))

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
#test_dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# TODO: check why tensorlfow logs are overlapping and non usable
# TODO: check how to name experiments for tracking 
# TODO: implement other metrics e.g. XNLI

print('Running experiment...')

training_args = TrainingArguments(
    run_name='experiments_distilBert_01',     # used for `wandb <https://www.wandb.com/>`_ logging.
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    learning_rate=2e-5,  #2e-5
    weight_decay=0.01,
    evaluation_strategy='epoch',
    logging_dir='./logs',
    logging_steps=500,
    save_strategy='no',
    report_to=['tensorboard'],
    # deepspeed='./ds_config.json'   # with cpu offload crashes, without it is slow in GoogleColab
    fp16=False,                      # without it is faster
    fp16_backend='auto', #'amp' , 'apex'
    disable_tqdm=True,
    load_best_model_at_end=True,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir logs