# !pip install -U sentence-transformers

import os
import random
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
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
import tensorflow_hub as hub
from numpy import dot
from numpy.linalg import norm


def cos_sim(a,b):
  return dot(a, b)/(norm(a)*norm(b))


def parse_argument():
	parser = argparse.ArgumentParser()
	parser.add_argument('--inter_data_path', type=str, default = './fsl_dataset/')
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--num_samples_per_class', type=int, default=1)
	parser.add_argument('--generalized', type=str, default="no")
	parser.add_argument('--encoder', type=str, default="USE")
	args = parser.parse_args()
	config = args.__dict__
	return config




if __name__ == "__main__":
	config = parse_argument()
	path = config["inter_data_path"]
	generalized = config["generalized"]
	encoder_type = config["encoder"]


	if generalized == "yes":
		test_tsv = pd.read_csv(path + "test_joint.tsv", sep='\t', header=None)
	else:
		test_tsv = pd.read_csv(path + "test_novel.tsv", sep="\t", header=None)

	print("---------------LOADING ENCODER-----------------")
	if encoder_type == "USE":
		encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
	elif encoder_type == "SBERT":
		encoder = SentenceTransformer('bert-base-nli-max-tokens')
	else:
		print("ERROR: Please choose encoder_type as USE or SBERT!")

	lablist = list(np.unique(test_tsv[0].to_list()))
	lab_to_emb = {}
	for lab in lablist:
		if encoder_type == "USE":
			lab_to_emb[lab] = encoder([lab])[0]
		elif encoder_type == "SBERT":
			lab_to_emb[lab] = encoder.encode([lab])[0]

	true_labels = test_tsv[0].to_list()
	pred_labels = []

	print("---------------RUNNING INFERENCE-----------------")
	for i in progressbar.progressbar(range(len(test_tsv[1]))):
  		instance = test_tsv[1].iloc[i]
  		if encoder_type == "USE":
  			instance_emb = encoder([instance])[0]
  		elif encoder_type == "SBERT":
  			instance_emb = encoder.encode([instance])[0]
  		label_to_score = {}
  		for lab in lablist:
  			label_to_score[lab] = cos_sim(lab_to_emb[lab], instance_emb)

  		#print(label_to_score)
  		sort_orders = sorted(label_to_score.items(), key=lambda x: x[1], reverse=True)
  		pred_labels.append(sort_orders[0][0])

	print(classification_report(true_labels, pred_labels, digits=4))