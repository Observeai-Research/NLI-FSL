# !pip install wget
# !pip install gitpython
import os
import random
import pandas as pd
import numpy as np
import csv
import wget
import shutil
import zipfile
import json
from git import Repo
import argparse


def parse_argument():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default = 'CLINC150')
	parser.add_argument('--inter_data_path', type=str, default = './fsl_dataset')
	parser.add_argument('--num_samples_per_class', type=int, default=1)
	args = parser.parse_args()
	config = args.__dict__
	return config


def clean_cache(direc="./cache/"):
	for files in os.listdir(direc):
		path = os.path.join(direc, files)
		try:
			shutil.rmtree(path)
		except OSError:
			os.remove(path)

	return




def helper(seen_labels, fsl_labels, length, label_to_utt):
	train_data = {}
	train_data["label"] = []
	train_data["utterance"] = []
	test_data = {}
	test_data["label"] = []
	test_data["utterance"] = []
	support1_seen = {}
	support1_seen["label"] = []
	support1_seen["utterance"] = []
	support5_seen = {}
	support5_seen["label"] = []
	support5_seen["utterance"] = []

	for sl in seen_labels:
		imp_utts = label_to_utt[sl]
		random.seed(30)
		imp_utts = random.sample(imp_utts, len(imp_utts))
		labels = len(imp_utts)*[sl]
		size_train = int(2*len(imp_utts)/3)
		size_test = len(imp_utts) - size_train
		train_data["label"].extend(labels[:size_train])
		train_data["utterance"].extend(imp_utts[:size_train])
		test_data["label"].extend(labels[size_train:])
		test_data["utterance"].extend(imp_utts[size_train:])
		random.seed(30)
		shot1_utts = random.sample(imp_utts[:size_train], 1)
		shot1_labels = 1*[sl]
		support1_seen["label"].extend(shot1_labels)
		support1_seen["utterance"].extend(shot1_utts)
		random.seed(30)
		shot5_utts = random.sample(imp_utts[:size_train], 5)
		shot5_labels = 5*[sl]
		support5_seen["label"].extend(shot5_labels)
		support5_seen["utterance"].extend(shot5_utts)

	test_data_novel = {}
	test_data_novel["label"] = []
	test_data_novel["utterance"] = []
	support1_novel = {}
	support1_novel["label"] = []
	support1_novel["utterance"] = []
	support5_novel = {}
	support5_novel["label"] = []
	support5_novel["utterance"] = []

	for lab in fsl_labels:
		imp_utts = label_to_utt[lab]
		random.seed(30)
		imp_utts = random.sample(imp_utts, len(imp_utts))
		labels = len(imp_utts)*[lab]
		size_train = int(2*len(imp_utts)/3)
		size_test = len(imp_utts) - size_train
		test_data_novel["label"].extend(labels[size_train:])
		test_data_novel["utterance"].extend(imp_utts[size_train:])
		random.seed(30)
		shot1_utts = random.sample(imp_utts[:size_train], 1)
		shot1_labels = 1*[lab]
		support1_novel["label"].extend(shot1_labels)
		support1_novel["utterance"].extend(shot1_utts)
		random.seed(30)
		shot5_utts = random.sample(imp_utts[:size_train], 5)
		shot5_labels = 5*[lab]
		support5_novel["label"].extend(shot5_labels)
		support5_novel["utterance"].extend(shot5_utts)

	train_data = pd.DataFrame.from_dict(train_data)
	test_data_novel = pd.DataFrame.from_dict(test_data_novel)
	test_data_seen = pd.DataFrame.from_dict(test_data)
	test_data_joint = pd.concat([test_data_seen, test_data_novel])

	support1_novel = pd.DataFrame.from_dict(support1_novel)
	support5_novel = pd.DataFrame.from_dict(support5_novel)
	support1_seen = pd.DataFrame.from_dict(support1_seen)
	support5_seen = pd.DataFrame.from_dict(support5_seen)
	support1_joint = pd.concat([support1_novel, support1_seen])
	support5_joint = pd.concat([support5_novel, support5_seen])

	if inter_data_path[-1] == "/":
		os.makedirs(inter_data_path + str(length) + "_percent/")
	else:
		os.makedirs(inter_data_path + "/" + str(length) + "_percent/")

	train_data.to_csv(inter_data_path + str(length) + "_percent/" + "train_seen.tsv", sep="\t", index=False, header=False)
	test_data_novel.to_csv(inter_data_path + str(length) + "_percent/" + "test_novel.tsv", sep="\t", index=False, header=False)
	test_data_joint.to_csv(inter_data_path + str(length) + "_percent/" + "test_joint.tsv", sep="\t", index=False, header=False)
	support1_novel.to_csv(inter_data_path + str(length) + "_percent/" + "support_1_shots_novel", sep="\t", index=False, header=False)
	support1_seen.to_csv(inter_data_path + str(length) + "_percent/" + "support_1_shots_seen", sep="\t", index=False, header=False)
	support1_joint.to_csv(inter_data_path + str(length) + "_percent/" + "support_1_shots_joint", sep="\t", index=False, header=False)
	support5_novel.to_csv(inter_data_path + str(length) + "_percent/" + "support_5_shots_novel", sep="\t", index=False, header=False)
	support5_seen.to_csv(inter_data_path + str(length) + "_percent/" + "support_5_shots_seen", sep="\t", index=False, header=False)
	support5_joint.to_csv(inter_data_path + str(length) + "_percent/" + "support_5_shots_joint", sep="\t", index=False, header=False)


def process_CLINC150(inter_data_path, num_shots, config):
	clean_cache()
	url='https://archive.ics.uci.edu/ml/machine-learning-databases/00570/clinc150_uci.zip'
	wget.download(url)
	shutil.move("./clinc150_uci.zip", "./cache/clinc150_uci.zip")
	with zipfile.ZipFile("./cache/clinc150_uci.zip", 'r') as zip_ref:
		zip_ref.extractall("./cache/")

	path = "./cache/clinc150_uci/data_full.json"
	f = open(path)
	data = json.load(f)
	labels = []
	utts = []
	for eg in data['val']:
		labels.append(eg[1])
		utts.append(eg[0])

	for eg in data['test']:
		labels.append(eg[1])
		utts.append(eg[0])

	for eg in data['train']:
		labels.append(eg[1])
		utts.append(eg[0])

	utt_to_label = {}
	for i,utt in enumerate(utts):
		utt_to_label[utt] = labels[i]

	all_labels = list(np.unique(labels))
	random.seed(30)
	fsl_labels = random.sample(all_labels, 50)
	seen_labels = []

	for l in all_labels:
		if l not in fsl_labels:
			seen_labels.append(l)

	label_to_utt = {}
	for l in all_labels:
		label_to_utt[l] = []

	for key in label_to_utt.keys():
		for utt in utt_to_label.keys():
			if utt_to_label[utt] == key:
				label_to_utt[key].append(utt)

	for_fsl = list(range(5,55,5))
	index = 0
	for length in range(10,110,10):
		this_set_seen_labels = seen_labels[:length]
		this_set_fsl_labels = fsl_labels[:for_fsl[index]]
		index = index + 1
		helper(seen_labels, fsl_labels, length, label_to_utt)
	return







def process_BANKING77(inter_data_path, num_shots, config):
	clean_cache()
	Repo.clone_from("https://github.com/PolyAI-LDN/task-specific-datasets.git", "./cache/")
	orig_train = pd.read_csv("./cache/banking_data/train.csv")
	orig_test = pd.read_csv("./cache/banking_data/test.csv")
	labels = []
	utts = []
	for i,eg in enumerate(orig_train["category"]):
		labels.append(eg)
		utts.append(orig_train["text"].iloc[i])

	for i,eg in enumerate(orig_test["category"]):
		labels.append(eg)
		utts.append(orig_test["text"].iloc[i])

	utt_to_label = {}
	for i,utt in enumerate(utts):
		utt_to_label[utt] = labels[i]

	all_labels = list(np.unique(labels))
	random.seed(30)
	fsl_labels = random.sample(all_labels, 27)
	seen_labels = []

	for l in all_labels:
		if l not in fsl_labels:
			seen_labels.append(l)

	label_to_utt = {}
	for l in all_labels:
		label_to_utt[l] = []

	for key in label_to_utt.keys():
		for utt in utt_to_label.keys():
			if utt_to_label[utt] == key:
				label_to_utt[key].append(utt)

	for_fsl = [2,5,8,10,13,16,18,21,24,27]
	index = 0
	for length in range(5,55,5):
		this_set_seen_labels = seen_labels[:length]
		this_set_fsl_labels = fsl_labels[:for_fsl[index]]
		index = index + 1
		helper(seen_labels, fsl_labels, length*2, label_to_utt)

	return







if __name__ == "__main__":
	config = parse_argument()
	inter_data_path = config["inter_data_path"]
	num_shots = config["num_samples_per_class"]
	dataset_type = config["dataset"]
	if not os.path.exists(inter_data_path):
	    os.makedirs(inter_data_path)
	if os.path.exists(inter_data_path):
		clean_cache(inter_data_path)

	if dataset_type == 'CLINC150':
		process_CLINC150(inter_data_path, num_shots, config)
	elif dataset_type == 'BANKING77':
		process_BANKING77(inter_data_path, num_shots, config)

	else:
		print("ERROR : Not a valid datatype. Choose from CLINC150 or BANKING77")