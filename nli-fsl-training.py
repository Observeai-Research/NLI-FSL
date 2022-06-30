import os
import random
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split
import textwrap
import progressbar
import keras
import argparse
import time
import datetime
import json
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertTokenizer
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report


def parse_argument():
	parser = argparse.ArgumentParser()
	parser.add_argument('--inter_data_path', type=str, default = './fsl_dataset/')
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--num_samples_per_class', type=int, default=1)
	parser.add_argument('--learning_rate', type=float, default=2e-6)
	parser.add_argument('--epochs', type=int, default=3)
	parser.add_argument('--generalized', type=bool, default=False)
	parser.add_argument('--max_seq_len', type=int, default=64)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--save_folder_name', type=str, default="./ckpt/")
	parser.add_argument('--ratio', type=int, default=2)
	parser.add_argument('--seen_class_data', type=str, default="yes")
	parser.add_argument('--seed', type=int, default=21)
	args = parser.parse_args()
	config = args.__dict__
	return config


def nli_format_data(new_shot):
	label_list = list(np.unique(new_shot[0].to_list()))
	df = {}
	df["label"] = []
	df["sentence"] = []
	df["label_name"] = []

	for i,sentence in enumerate(new_shot[1]):
		for label in label_list:
			if new_shot[0].iloc[i] == label:
				df["label"].append(1)
			else:
				df["label"].append(0)

			df["sentence"].append(sentence)
			df["label_name"].append(label)

	return pd.DataFrame.from_dict(df)



def sampling_non_entailed(train, support, num_shots, config):
	train_df_0 = train[train['label'] == 0]
	train_df_1 = train[train['label'] == 1]
	train_df_0 = train_df_0.sample(n=min(config["ratio"]*len(train_df_1), len(train_df_0)), random_state=1)
	train_df = pd.concat([train_df_0, train_df_1])
	train_df = train_df.sample(n=len(train_df), random_state=1)
	if num_shots and config["seen_class_data"]=="yes":
		combined_df = pd.concat([train_df, support])
	elif num_shots and config["seen_class_data"]=="no":
		combined_df = pd.concat([support])
	else:
		combined_df = pd.concat([train_df])
	return combined_df



def att_masking(input_ids):
  attention_masks = []
  for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)
  return attention_masks




def input_id_maker(dataf, tokenizer, mapper, config, embtype="new"):
  input_ids = []
  tok_type_ids = []
  lengths = []

  for i in progressbar.progressbar(range(len(dataf["label_name"]))):
    sen = dataf["sentence"].iloc[i]
    sen = tokenizer.tokenize(sen)
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token

    label_name = mapper[dataf["label_name"].iloc[i]]
    #print(label_name)
    label_toks = tokenizer.tokenize(label_name)

    if(len(sen) + len(label_toks) > config["max_seq_len"]-3):
      print("happens")
      sen = sen[:config["max_seq_len"]-3-len(label_toks)]
    if embtype == "new":
      sen_new = [CLS] + sen + [SEP] + label_toks + [SEP]
      tok_type_id = (len(sen)+2)*[0] + (len(label_toks)+1)*[1]
      if len(tok_type_id) < config["max_seq_len"]:
        tok_type_id = tok_type_id + (config["max_seq_len"]-len(tok_type_id))*[1]
    else:
      sen_new = [CLS] + sen + [SEP]
      tok_type_id = []
    encoded_sent = tokenizer.convert_tokens_to_ids(sen_new)
    input_ids.append(encoded_sent)
    lengths.append(len(encoded_sent))
    tok_type_ids.append(tok_type_id)

  input_ids = pad_sequences(input_ids, maxlen=config["max_seq_len"], value=0, dtype="long", truncating="post", padding="post")
  return input_ids, tok_type_ids






def prepare_train_data(train_tsv, config, tokenizer, mapper):
	train_input_ids, train_lengths = input_id_maker(train_tsv, tokenizer, mapper, config)
	validation_input_ids, validation_lengths = input_id_maker(train_tsv[:int(len(train_tsv)/9)], tokenizer, mapper, config)

	train_attention_masks = att_masking(train_input_ids)
	validation_attention_masks = att_masking(validation_input_ids)

	train_labels = train_tsv["label"].to_numpy().astype('int')
	validation_labels = train_tsv[:int(len(train_tsv)/9)]["label"].to_numpy().astype('int')

	train_inputs = train_input_ids
	validation_inputs = validation_input_ids
	train_masks = train_attention_masks
	validation_masks = validation_attention_masks

	train_inputs = torch.tensor(train_inputs)
	train_labels = torch.tensor(train_labels)
	train_masks = torch.tensor(train_masks)
	validation_inputs = torch.tensor(validation_inputs)
	validation_labels = torch.tensor(validation_labels)
	validation_masks = torch.tensor(validation_masks)

	batch_size = config["batch_size"]
	train_data = TensorDataset(train_inputs, train_masks, train_labels)
	train_sampler = RandomSampler(train_data)
	train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size = batch_size)
	validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
	validation_sampler = RandomSampler(validation_data)
	validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size = batch_size)

	return train_data, train_sampler, train_dataloader, validation_data, validation_sampler, validation_dataloader




def load_model_and_tokenizer(config):
	model_type = 'bert'
	model_class, tokenizer_class, config_class = BertForSequenceClassification, BertTokenizer, BertConfig
	model_name = 'bert-base-uncased'
	tokenizer = tokenizer_class.from_pretrained(model_name)
	device = config["device"]
	model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
	model.to(config["device"])
	return model, tokenizer






def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



def train(config, model, tokenizer, train_data, train_sampler, train_dataloader, validation_data, validation_sampler, validation_dataloader):
	lr = config["learning_rate"]
	device = config["device"]
	max_grad_norm = 1.0
	epochs = config["epochs"]
	num_total_steps = len(train_dataloader)*epochs
	num_warmup_steps = 1000
	warmup_proportion = float(num_warmup_steps) / float(num_total_steps)  # 0.1
	optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = num_total_steps)

	seed_val = config["seed"]

	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed_all(seed_val)

	loss_values = []

	# For each epoch...
	for epoch_i in range(0, epochs):
	    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
	    print('Training...')

	    t0 = time.time()
	    total_loss = 0

	    model.train()

	    for step, batch in enumerate(train_dataloader):
	        if step % 40 == 0 and not step == 0:
	            print('  Batch {:>5,}  of  {:>5,}. '.format(step, len(train_dataloader)))

	        
	        b_input_ids = batch[0].to(device)
	        b_input_mask = batch[1].to(device)
	        b_labels = batch[2].to(device)

	        model.zero_grad()        

	        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
	        
	        loss = outputs[0]
	        total_loss += loss.item()
	        loss.backward()

	        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

	        optimizer.step()
	        scheduler.step()

	    avg_train_loss = total_loss / len(train_dataloader)            
	    loss_values.append(avg_train_loss)

	    print("")
	    print("  Average training loss: {0:.2f}".format(avg_train_loss))
	        
	    print("")
	    print("Running Validation...")

	    t0 = time.time()

	    model.eval()

	    eval_loss, eval_accuracy = 0, 0
	    nb_eval_steps, nb_eval_examples = 0, 0

	    for batch in validation_dataloader:
	        batch = tuple(t.to(device) for t in batch)
	        b_input_ids, b_input_mask, b_labels = batch
	        
	        with torch.no_grad():        
	          outputs = model(b_input_ids, attention_mask=b_input_mask)
	    
	        logits = outputs[0]

	        logits = logits.detach().cpu().numpy()
	        label_ids = b_labels.to('cpu').numpy()
	        
	        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
	        eval_accuracy += tmp_eval_accuracy

	        nb_eval_steps += 1

	    # Report the final accuracy for this validation run.
	    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))

	print("")
	print("Training complete!")
	return model, tokenizer




if __name__ == "__main__": 
	# load settings
	config = parse_argument()
	path = config["inter_data_path"]
	num_shots = config["num_samples_per_class"]

	print("--------------READING FEW-SHOT DATASET---------------")
	train_tsv = pd.read_csv(path + 'train_seen.tsv', sep='\t', header=None)
	if num_shots:
		new_shot = pd.read_csv(path + "support_" + str(num_shots) + "_shots_novel", sep='\t', header=None)

	train_tsv = train_tsv.dropna()

	print("--------------LOADING BERT MODEL---------------\n")
	model, tokenizer = load_model_and_tokenizer(config)

	print("--------------PREPARING NLI FORMAT TRAIN DATA---------------\n")
	desc = {}
	if num_shots:
		desc["labels"] = list(np.unique(list(np.unique(train_tsv[0].to_list())) + list(np.unique(new_shot[0].to_list()))))
	else:
		desc["labels"] = list(np.unique(list(np.unique(train_tsv[0].to_list()))))

	mapper = {}
	for i,label in enumerate(desc["labels"]):
  		mapper[label] = desc["labels"][i]

	nli_new_shot = ""
	if num_shots:
		nli_new_shot = nli_format_data(new_shot)
	nli_train_df = nli_format_data(train_tsv)

	combined_df = sampling_non_entailed(nli_train_df, nli_new_shot, num_shots, config)

	train_data, train_sampler, train_dataloader, validation_data, validation_sampler, validation_dataloader = prepare_train_data(combined_df, config, tokenizer, mapper)

	print("----------------STARTING TRAINING------------------\n\n")
	model, tokenizer = train(config, model, tokenizer, train_data, train_sampler, train_dataloader, validation_data, validation_sampler, validation_dataloader)

	print("----------------SAVING MODEL----------------------\n")

	output_dir = config["save_folder_name"]
	if not os.path.exists(output_dir):
	    os.makedirs(output_dir)
	print("Saving model to %s" % output_dir)
	model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
	model_to_save.save_pretrained(output_dir)
	tokenizer.save_pretrained(output_dir)

