# needs transformers
# path to the intermediate dataset folder: inter_path
# generalized few shot learning parameter: generalized = True
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
import time
import datetime
import json
import argparse
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertTokenizer
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report



def parse_argument():
	parser = argparse.ArgumentParser()
	parser.add_argument('--inter_data_path', type=str, default = './fsl_dataset/')
	parser.add_argument('--model_path', type=str, default = './ckpt/')
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--max_seq_len', type=int, default=64)
	parser.add_argument('--num_samples_per_class', type=int, default=1)
	parser.add_argument('--generalized', type=str, default="No")
	parser.add_argument('--batch_size', type=int, default=32)
	args = parser.parse_args()
	config = args.__dict__
	return config


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

  for i in range(len(dataf["label_name"])):
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





def load_model_and_tokenizer(config, saved_model_path, num_labels_in_seen_class):
	tokenizer = BertTokenizer.from_pretrained(saved_model_path)
	device = config["device"]
	model = BertForSequenceClassification.from_pretrained(saved_model_path, output_hidden_states=True)
	model.to(device)
	return model, tokenizer





def nli_predictions(model, tokenizer, test_tsv, label_dict_test, labellist_test, config, mapper):
	pred_labels = []
	true_labels = []

	model.eval()

	for i in progressbar.progressbar(range(len(test_tsv[1]))):
		sentence = test_tsv[1].iloc[i]
		true_labels.append(label_dict_test[test_tsv[0].iloc[i]])
		df_dummy = {}
		df_dummy["sentence"] = []
		df_dummy["label_name"] = []
		df_dummy["label"] = []
		for label in labellist_test:
			#print(sentence)
			df_dummy["sentence"].append(sentence)
			df_dummy["label_name"].append(label)
			df_dummy["label"].append(label_dict_test[label])

		df_dummy = pd.DataFrame.from_dict(df_dummy)
		test_input_ids, test_tok_ids = input_id_maker(df_dummy, tokenizer, mapper, config)
		test_attention_masks = att_masking(test_input_ids)
		test_labels = df_dummy["label"].to_numpy().astype('int')
		test_inputs = torch.tensor(test_input_ids)
		test_tti = torch.tensor(test_tok_ids)
		test_masks = torch.tensor(test_attention_masks)
		test_labels = torch.tensor(test_labels)
		prediction_data = TensorDataset(test_inputs, test_masks, test_tti, test_labels)
		prediction_sampler = SequentialSampler(prediction_data)
		prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=config["batch_size"])
		this_sent_preds = {}
		for (step, batch) in enumerate(prediction_dataloader):
			batch = tuple(t.to(config["device"]) for t in batch)
			b_input_ids, b_input_mask, b_tti, b_labels = batch
			with torch.no_grad():
			  outputs = model(b_input_ids, token_type_ids=b_tti, attention_mask=b_input_mask)

			logits = outputs[0]
			logits = logits.detach().cpu().numpy()
			b_labels = b_labels.detach().cpu().numpy()
			for j,pair in enumerate(logits):
			  probability = np.exp(pair)/sum(np.exp(pair))
			  this_sent_preds[b_labels[j]] = probability[0]

		sort_orders = sorted(this_sent_preds.items(), key=lambda x: x[1], reverse=False)
		pred_labels.append(sort_orders[0][0])

	return pred_labels, true_labels



if __name__ == "__main__": 
	# load settings
	config = parse_argument()
	path = config["inter_data_path"]
	train_tsv = pd.read_csv(path + "train_seen.tsv", sep='\t', header=None)
	generalized = config["generalized"]
	shots = config["num_samples_per_class"]
	novel_df = pd.read_csv(path + "support_" + str(shots) + "_shots_novel", sep="\t", header=None)

	print("--------------READING FEW-SHOT DATASET---------------")

	if generalized == "yes":
		test_tsv = pd.read_csv(path + "test_joint.tsv", sep='\t', header=None)
	else:
		test_tsv = pd.read_csv(path + "test_novel.tsv", sep="\t", header=None)

	train_tsv = train_tsv.dropna()
	test_tsv = test_tsv.dropna()

	desc = {}
	desc["labels"] = list(np.unique(list(np.unique(train_tsv[0].to_list())) + list(np.unique(novel_df[0].to_list()))))

	mapper = {}
	for i,label in enumerate(desc["labels"]):
  		mapper[label] = desc["labels"][i]

	print("--------------LOADING BERT MODEL FROM CHECKPOINT---------------\n")

	model, tokenizer = load_model_and_tokenizer(config, config["model_path"], 2)

	new_shot = ""
	if generalized != "yes":
		new_shot = pd.read_csv(path + "support_" + str(shots) + "_shots_novel", sep='\t', header=None)
	else:
		new_shot = pd.read_csv(path + "support_" + str(shots) + "_shots_joint", sep='\t', header=None)

	labellist_test = list(np.unique(new_shot[0].to_list()))

	label_dict_test = {}
	for i,l in enumerate(labellist_test):
	  label_dict_test[l] = i

	print(label_dict_test)
	print("--------------GETTING PREDICTIONS---------------\n")
	preds, trues = nli_predictions(model, tokenizer, test_tsv, label_dict_test, labellist_test, config, mapper)
	print(classification_report(trues, preds, digits=4))