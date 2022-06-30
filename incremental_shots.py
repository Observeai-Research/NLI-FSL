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
	parser.add_argument('--inter_data_path', type=str, default = './fsl_dataset')
	parser.add_argument('--num_samples_per_class_list', type=str, default='1 5 10 20')
	args = parser.parse_args()
	config = args.__dict__
	return config


def clean_cache(direc="./cache/"):
	if os.path.exists("./cache/"):
		for files in os.listdir(direc):
			path = os.path.join(direc, files)
			try:
				shutil.rmtree(path)
			except OSError:
				os.remove(path)
	else:
		os.makedirs("./cache/")

	return




def helper(seen_labels, fsl_labels, num_shots, label_to_utt):
	train_data = {}
	train_data["label"] = []
	train_data["utterance"] = []
	test_data = {}
	test_data["label"] = []
	test_data["utterance"] = []
	support_seen = {}
	support_seen["label"] = []
	support_seen["utterance"] = []

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
		shot_utts = random.sample(imp_utts[:size_train], num_shots)
		shot_labels = num_shots*[sl]
		support_seen["label"].extend(shot_labels)
		support_seen["utterance"].extend(shot_utts)

	test_data_novel = {}
	test_data_novel["label"] = []
	test_data_novel["utterance"] = []
	support_novel = {}
	support_novel["label"] = []
	support_novel["utterance"] = []

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
		shot_utts = random.sample(imp_utts[:size_train], num_shots)
		shot_labels = num_shots*[lab]
		support_novel["label"].extend(shot_labels)
		support_novel["utterance"].extend(shot_utts)

	train_data = pd.DataFrame.from_dict(train_data)
	test_data_novel = pd.DataFrame.from_dict(test_data_novel)
	test_data_seen = pd.DataFrame.from_dict(test_data)
	test_data_joint = pd.concat([test_data_seen, test_data_novel])

	support_novel = pd.DataFrame.from_dict(support_novel)
	support_seen = pd.DataFrame.from_dict(support_seen)
	support_joint = pd.concat([support_novel, support_seen])

	if inter_data_path[-1] == "/":
		os.makedirs(inter_data_path + str(num_shots) + "_shots/")
	else:
		os.makedirs(inter_data_path + "/" + str(num_shots) + "_shots/")

	train_data.to_csv(inter_data_path + str(num_shots) + "_shots/" + "train_seen.tsv", sep="\t", index=False, header=False)
	test_data_novel.to_csv(inter_data_path + str(num_shots) + "_shots/" + "test_novel.tsv", sep="\t", index=False, header=False)
	test_data_joint.to_csv(inter_data_path + str(num_shots) + "_shots/" + "test_joint.tsv", sep="\t", index=False, header=False)
	support_novel.to_csv(inter_data_path + str(num_shots) + "_shots/" + "support_" + str(num_shots) + "_shots_novel", sep="\t", index=False, header=False)
	support_seen.to_csv(inter_data_path + str(num_shots) + "_shots/" + "support_" + str(num_shots) + "_shots_seen", sep="\t", index=False, header=False)
	support_joint.to_csv(inter_data_path + str(num_shots) + "_shots/" + "support_" + str(num_shots) + "_shots_joint", sep="\t", index=False, header=False)


def process_CLINC150(inter_data_path, num_shots_list):
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


	for num_shots in num_shots_list:
		helper(seen_labels, fsl_labels, num_shots, label_to_utt)
	return








if __name__ == "__main__":
	config = parse_argument()
	inter_data_path = config["inter_data_path"]
	num_shots_str = config["num_samples_per_class_list"]
	num_shots_list = list(num_shots_str.split(" "))
	num_shots_list = [int(val) for val in num_shots_list]
	if not os.path.exists(inter_data_path):
	    os.makedirs(inter_data_path)
	if os.path.exists(inter_data_path):
		clean_cache(inter_data_path)
	
	process_CLINC150(inter_data_path, num_shots_list)