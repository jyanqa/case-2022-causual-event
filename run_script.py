import torch
import datasets
from collections import Counter
from datasets import load_dataset
from datasets import Dataset
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
from torch import nn
from transformers import Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers.optimization import Adafactor, AdafactorSchedule
from transformers import AdamW, get_linear_schedule_with_warmup
import argparse

def main():
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_train = pd.read_csv('data/ST1/train_subtask1.csv')
    df_val = pd.read_csv('data/ST1/dev_subtask1.csv')
    df_test = pd.read_csv('data/ST1/test_subtask1_text.csv')

    df_trainval = pd.concat([df_train,df_val])
    df_test['label'] = 0
    df_test['agreement'] = 0
    df_test['num_votes'] = 0

    train_ds = Dataset.from_pandas(df_train)
    val_ds = Dataset.from_pandas(df_val)
    trainval_ds = Dataset.from_pandas(df_trainval)
    test_ds = Dataset.from_pandas(df_test)

    df_train['ce'] = df_train['agreement'] # add new column 
    df_train['non-ce'] = df_train['agreement']
    for i in range(df_train.shape[0]): 
        if df_train['label'][i] == 1:
            df_train['ce'][i] = df_train['agreement'][i] 
            df_train['non-ce'][i] = (1 - df_train['agreement'][i])
        else:
            df_train['non-ce'][i] = df_train['agreement'][i]
            df_train['ce'][i] = (1 - df_train['agreement'][i])
        
    # Load BERT/ROBERTA/XLNet tokenizer.
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    def encode_dataset(dataset: datasets.arrow_dataset.Dataset) -> list:
        '''
        Transforming each instance of the dataset with the Tokenizer
        '''
        encoded_dataset = []
        for item in dataset:
            # Tokenize the sentence.
            sentence_encoded = tokenizer(item['text'],
                                        return_tensors="pt", 
                                        padding='max_length', 
                                        truncation=True, 
                                        max_length=60) 
            
            sentence_encoded['labels'] = torch.LongTensor(np.array([item['label']]))
            sentence_encoded['num_votes'] = torch.LongTensor(np.array(np.around([item['num_votes']],3))) #number of vote
            sentence_encoded['agreement'] = torch.Tensor(np.array([item['agreement']])) #agreement
            encoded_dataset.append(sentence_encoded)

        # Reduce dimensionality of tensors.
        for item in encoded_dataset:
            for key in item:
                item[key] = torch.squeeze(item[key])
        return encoded_dataset
    
    # Tokenizing datasets
    encoded_dataset_train = encode_dataset(train_ds)
    encoded_dataset_val = encode_dataset(val_ds)
    encoded_dataset_trainval = encode_dataset(trainval_ds)
    encoded_dataset_test = encode_dataset(test_ds)

    # Create dictionaries to transform from labels to id and vice-versa.
    id2label = {0 : 'causualeffect',
                1 : 'non-causualeffect'}
    label2id = {v:k for k,v in id2label.items()}
    num_labels = len(label2id)


    class CustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False):
            y_true = inputs.get("labels")
            n = inputs.get('num_votes') #number of votes
            r = inputs.get('agreement') #agreement
            # forward pass 
            inputs2 = {"input_ids":inputs.get("input_ids"), "labels":inputs.get("labels"),
                    "attention_mask":inputs.get("attention_mask"),
                    "token_type_ids":inputs.get("token_type_ids")}
            outputs = model(**inputs2)
            logits = outputs.get("logits")
            y_pred = torch.softmax(logits,dim=1)[:,1]
            # compute custom loss  #todo: 
            if args.loss_name == 'ce':
                loss = torch.mean(-y_true*torch.log(y_pred) - (1-y_true)*torch.log(1-y_pred))
            elif args.loss_name == 'ce2':
                loss1 = n*r*torch.log(y_pred) + n*(1-r)*torch.log(1-y_pred) #if y_true = 1
                loss2 = n*r*torch.log(1-y_pred) + n*(1-r)*torch.log(y_pred) #if y_true = 0
                loss = -torch.sum(y_true*loss1+(1-y_true)*loss2)
                loss = loss/len(torch.sum(n))
            elif args.loss_name == 'ce3':
                loss1 = n*r*torch.log(y_pred)  #if y_true = 1
                loss2 = n*r*torch.log(1-y_pred)  #if y_true = 0
                loss = -torch.sum(y_true*loss1+(1-y_true)*loss2)
                loss = loss/len(torch.sum(n*r))
            return (loss, outputs) if return_outputs else loss
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
    
    training_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir='logs',
        no_cuda=False,  
        output_dir ='.',
        seed = 42,
        learning_rate = args.learning_rate, #defaults 1e-3
        warmup_steps=0, # number of warmup steps for learning rate scheduler
        weight_decay=0.001,
        evaluation_strategy='steps', #defaults: 'no'
        load_best_model_at_end = True,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels,ignore_mismatched_sizes=True) 
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=encoded_dataset_train,
        eval_dataset=encoded_dataset_val,
        compute_metrics=compute_metrics,
        )
    # Fine tunning
    trainer.train()
    trainer.evaluate()


    preds_val = trainer.predict(encoded_dataset_val)
    predictions = preds_val.predictions.argmax(-1)

    # Create array with predicted labels and expected.
    true_values = np.array(preds_val.label_ids).flatten()
    predicted_values = np.array(preds_val.predictions.argmax(-1)).flatten()

    # Filter the labels. We only produce a label for each word. We filter labels
    # of subwords and special tokens, such as PAD
    proc_predicted_values = [prediction for prediction, label in zip(predicted_values, true_values) if label != -100]
    proc_true_values = [label for prediction, label in zip(predicted_values, true_values) if label != -100]

    # Evaluate models
    model_performance = {}
    model_performance['accuracy'] = accuracy_score(proc_true_values, proc_predicted_values)
    model_performance['precision_micro'] = precision_score(proc_true_values, proc_predicted_values, average='micro')
    model_performance['precision_macro'] = precision_score(proc_true_values, proc_predicted_values, average='macro')
    model_performance['recall_micro'] = recall_score(proc_true_values, proc_predicted_values, average='micro')
    model_performance['recall_macro'] = recall_score(proc_true_values, proc_predicted_values, average='macro')
    model_performance['f1_micro'] = f1_score(proc_true_values, proc_predicted_values, average='micro')
    model_performance['f1_macro'] = f1_score(proc_true_values, proc_predicted_values, average='macro')
    model_performance["f1_binary"] = f1_score(proc_true_values, proc_predicted_values,)
    model_performance['confusion_matrix'] = confusion_matrix(proc_true_values, proc_predicted_values)
    model_performance['confusion_matrix_normalized'] = confusion_matrix(proc_true_values, proc_predicted_values, normalize='true')
    # print('------------Model performance------------')
    # print(f'  recall_micro: {model_performance["recall_micro"]}')
    # print(f'  recall_macro: {model_performance["recall_macro"]}')
    # print(f'  accuracy: {model_performance["accuracy"]}')
    print(f'  precision_micro: {model_performance["precision_micro"]}')
    print(f'  precision_macro: {model_performance["precision_macro"]}')
    print(f'  f1_binary: {model_performance["f1_binary"]}')
    print(f'  f1-micro: {model_performance["f1_micro"]}')
    print(f'  f1-macro: {model_performance["f1_macro"]}')

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('-n','--num_train_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=82)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--logging_dir', type=str, default='logs')
    parser.add_argument('--evaluation_strategy', type=str, default='steps')
    parser.add_argument("-l", "--loss_name", type=str, default="ce", choices=["ce", "ce2", "ce3"])
    return parser.parse_args()

if __name__ == '__main__':
    main()
