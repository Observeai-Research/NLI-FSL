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
	parser.add_argument('--max_seq_len', type=int, default=48)
	parser.add_argument('--num_samples_per_class', type=int, default=1)
	parser.add_argument('--generalized', type=str, default="no")
	parser.add_argument('--use_saved_model', type=str, default="yes")
	args = parser.parse_args()
	config = args.__dict__
	return config



def att_masking(input_ids):
  attention_masks = []
  for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)
  return attention_masks




def get_output_for_one_vec(input_id, att_mask, device):
  input_ids = torch.tensor(input_id)
  att_masks = torch.tensor(att_mask)
  input_ids = input_ids.unsqueeze(0)
  att_masks = att_masks.unsqueeze(0)
  model.eval()
  input_ids = input_ids.to(device)
  att_masks = att_masks.to(device)
  with torch.no_grad():
      hidden_states = model(input_ids=input_ids, token_type_ids=None, attention_mask=att_masks).hidden_states

  vec = hidden_states[12][0][0]
  vec = vec.detach().cpu().numpy()
  return vec


def generate_np_files_for_emb(dataf, tokenizer, max_seq_len, device):
  input_ids, lengths = input_id_maker(dataf, tokenizer, max_seq_len)
  vecs = []
  att_masks = att_masking(input_ids)
  for index,ii in enumerate(input_ids):
  	vecs.append(get_output_for_one_vec(ii, att_masks[index], device))

  return vecs



def input_id_maker(dataf, tokenizer, max_seq_len):
  input_ids = []
  lengths = []

  for i in range(len(dataf[1])):
    sen = dataf[1].iloc[i]
    sen = tokenizer.tokenize(sen)
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    if(len(sen) > max_seq_len):
      print("happens")
      sen = sen[len(sen)-max_seq_len:]
      
    sen = [CLS] + sen + [SEP]
    encoded_sent = tokenizer.convert_tokens_to_ids(sen)
    input_ids.append(encoded_sent)
    lengths.append(len(encoded_sent))

  input_ids = pad_sequences(input_ids, maxlen=max_seq_len, value=0, dtype="long", truncating="post", padding="post")
  return input_ids, lengths


# num labels in seen class needed for below function


def load_model_and_tokenizer(config, saved_model_path, saved_model, num_labels_in_seen_class):
	if saved_model == "yes":
		tokenizer = BertTokenizer.from_pretrained(saved_model_path)
		device = config["device"]
		model = BertForSequenceClassification.from_pretrained(saved_model_path, output_hidden_states=True)
		model.to(device)
		return model, tokenizer
	else:
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		device = config["device"]
		model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_hidden_states=True)
		model.to(device)
		return model, tokenizer







if __name__ == "__main__": 
	# load settings
	config = parse_argument()
	path = config["inter_data_path"]

	train_tsv = pd.read_csv(path + "train_seen.tsv", sep='\t', header=None)
	generalized = config["generalized"]
	shots = config["num_samples_per_class"]

	print("--------------READING FEW-SHOT DATASET---------------")

	if generalized == "yes":
		test_tsv = pd.read_csv(path + "test_joint.tsv", sep='\t', header=None)
	else:
		test_tsv = pd.read_csv(path + "test_novel.tsv", sep="\t", header=None)

	train_tsv = train_tsv.dropna()
	test_tsv = test_tsv.dropna()

	num_seen_class_labels = len(np.unique(train_tsv[0].to_list()))

	print("--------------LOADING BERT MODEL FROM CHECKPOINT---------------\n")

	model, tokenizer = load_model_and_tokenizer(config, config["model_path"], config["use_saved_model"], num_seen_class_labels)

	print("--------------INFERENCE-------------------------------------------------------------\n\n")

	# novel data loading
	if generalized == "yes":
		new_shot = pd.read_csv(path + "support_" + str(shots) + "_shots_joint", sep='\t', header=None)
	else:
		new_shot = pd.read_csv(path + "support_" + str(shots) + "_shots_novel", sep='\t', header=None)

	print("-----------------OBTAINING TEST EMBEDDINGS-------------------------------------------\n")

	max_seq_len = config["max_seq_len"]
	device = config["device"]
	test_embeddings = generate_np_files_for_emb(test_tsv, tokenizer, max_seq_len, device)

	print("----------------OBTAINING SUPPORT PROTOTYPES------------------------------------------\n")

	new_shot = new_shot.sort_values(0)
	label_prototypes = {}
	labellist_test = list(np.unique(new_shot[0].to_list()))

	label_dict_test = {}
	for i,l in enumerate(labellist_test):
	  label_dict_test[l] = i

	print(label_dict_test)

	embset = {}
	embset[str(shots) + "shot"] = []
	j = 0
	for i in range(len(labellist_test)):
  		embset[str(shots) + "shot"].append(generate_np_files_for_emb(new_shot[j:j+shots], tokenizer, max_seq_len, device))
  		j = j+shots

	label_prototypes[str(shots) + "shot"] = {}
	for i in range(len(labellist_test)):
	  label_prototypes[str(shots) + "shot"][i] = np.mean(np.array(embset[str(shots) + "shot"][i]), axis=0)

	#print(label_prototypes)

	true_labels = test_tsv[0].to_list()
	true_labels = [label_dict_test[l] for l in true_labels]
	print(true_labels)

	test_embs = list(test_embeddings)
	pred_labels = []
	for i,emb in enumerate(test_embeddings):
	  dist_list = {}
	  for i,l in enumerate(labellist_test):
	    dist_list[label_dict_test[l]] = np.linalg.norm(label_prototypes[str(shots) + "shot"][i] - emb)
	  
	  sort_orders = sorted(dist_list.items(), key=lambda x: x[1], reverse=False)
	  pred_labels.append(sort_orders[0][0])


	print("----------------REPORT------------------------------------------\n\n\n")
	print(classification_report(true_labels, pred_labels, digits=4))