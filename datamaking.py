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
	parser.add_argument('--dataset', type=str, default = 'SNIPS')
	parser.add_argument('--inter_data_path', type=str, default = './fsl_dataset')
	parser.add_argument('--percent_novel_classes', type=int, default = 33)
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




def process_SNIPS(inter_data_path, num_shots):
	clean_cache()
	novel_test_tsv = pd.read_csv("./SMAN/SMAN_raw_data/SNIPS/test_novel.tsv", sep="\t", header=None)
	joint_test_tsv = pd.read_csv("./SMAN/SMAN_raw_data/SNIPS/test_joint.tsv", sep="\t", header=None)
	train_tsv = pd.read_csv("./SMAN/SMAN_raw_data/SNIPS/train_seen.tsv", sep="\t", header=None)

	new_train_tsv = {}
	new_train_tsv[0] = []
	new_train_tsv[1] = []
	for i, label in enumerate(train_tsv[0]):
		if label == "play music":
			continue
		else:
			new_train_tsv[0].append(label)
			new_train_tsv[1].append(train_tsv[1].iloc[i])

	new_train_tsv = pd.DataFrame.from_dict(new_train_tsv)

	new_novel_test_tsv = {}
	new_novel_test_tsv[0] = []
	new_novel_test_tsv[1] = []

	play_music_sentences = []

	for i, label in enumerate(joint_test_tsv[0]):
		if label == "play music":
			play_music_sentences.append(joint_test_tsv[1].iloc[i])
		else:
			continue

	for i, sentence in enumerate(play_music_sentences):
	  new_novel_test_tsv[0].append("play music")
	  new_novel_test_tsv[1].append(sentence)

	new_novel_test_tsv = pd.DataFrame.from_dict(new_novel_test_tsv)
	addplaylist_df = novel_test_tsv[:381]
	ratebook_df = novel_test_tsv[381:]

	final_novel_test_tsv = pd.concat([addplaylist_df, new_novel_test_tsv, ratebook_df])
	support_1shot_novel = pd.read_csv("./SMAN/SMAN_raw_data/SNIPS/support_1_shots_novel", sep="\t", header=None)
	support_1shot_joint = pd.read_csv("./SMAN/SMAN_raw_data/SNIPS/support_1_shots_joint", sep="\t", header=None)
	support_1shot_seen = pd.read_csv("./SMAN/SMAN_raw_data/SNIPS/support_1_shots_seen", sep="\t", header=None)
	support_5shot_novel = pd.read_csv("./SMAN/SMAN_raw_data/SNIPS/support_5_shots_novel", sep="\t", header=None)
	support_5shot_joint = pd.read_csv("./SMAN/SMAN_raw_data/SNIPS/support_5_shots_joint", sep="\t", header=None)
	support_5shot_seen = pd.read_csv("./SMAN/SMAN_raw_data/SNIPS/support_5_shots_seen", sep="\t", header=None)

	support_1shot_seen_upper = support_1shot_seen[:2]
	support_1shot_seen_lower = support_1shot_seen[3:]
	new_support_1shot_seen = pd.concat([support_1shot_seen_upper, support_1shot_seen_lower])

	support_5shot_seen_upper = support_5shot_seen[:10]
	support_5shot_seen_lower = support_5shot_seen[15:]
	new_support_5shot_seen = pd.concat([support_5shot_seen_upper, support_5shot_seen_lower])

	support_1shot_novel_upper = support_1shot_novel[:1]
	support_1shot_novel_lower = support_1shot_novel[1:]
	support_1shot_novel_middle = support_1shot_seen[2:3]
	new_support_1shot_novel = pd.concat([support_1shot_novel_upper, support_1shot_novel_middle, support_1shot_novel_lower])

	support_5shot_novel_upper = support_5shot_novel[:5]
	support_5shot_novel_lower = support_5shot_novel[5:]
	support_5shot_novel_middle = support_5shot_seen[10:15]
	new_support_5shot_novel = pd.concat([support_5shot_novel_upper, support_5shot_novel_middle, support_5shot_novel_lower])

	new_train_tsv.to_csv(inter_data_path + "train_seen.tsv", sep="\t", header=None, index=False)
	final_novel_test_tsv.to_csv(inter_data_path + "test_novel.tsv", sep="\t", header=None, index=False)
	joint_test_tsv.to_csv(inter_data_path + "test_joint.tsv", sep="\t", header=None, index=False)

	new_support_1shot_novel.to_csv(inter_data_path + "support_1_shots_novel", sep="\t", header=None, index=False)
	new_support_1shot_seen.to_csv(inter_data_path + "support_1_shots_seen", sep="\t", header=None, index=False)
	support_1shot_joint.to_csv(inter_data_path + "support_1_shots_joint", sep="\t", header=None, index=False)
	new_support_5shot_novel.to_csv(inter_data_path + "support_5_shots_novel", sep="\t", header=None, index=False)
	new_support_5shot_seen.to_csv(inter_data_path + "support_5_shots_seen", sep="\t", header=None, index=False)
	support_5shot_joint.to_csv(inter_data_path + "support_5_shots_joint", sep="\t", header=None, index=False)
	return

def process_NLUE(inter_data_path, num_shots):
	clean_cache()
	support_1shot_novel = pd.read_csv("./SMAN/SMAN_raw_data/NLUE/KFold_1/support_1_shots_novel", sep="\t", header=None)
	support_1shot_joint = pd.read_csv("./SMAN/SMAN_raw_data/NLUE/KFold_1/support_1_shots_joint", sep="\t", header=None)
	support_1shot_seen = pd.read_csv("./SMAN/SMAN_raw_data/NLUE/KFold_1/support_1_shots_seen", sep="\t", header=None)
	support_5shot_novel = pd.read_csv("./SMAN/SMAN_raw_data/NLUE/KFold_1/support_5_shots_novel", sep="\t", header=None)
	support_5shot_joint = pd.read_csv("./SMAN/SMAN_raw_data/NLUE/KFold_1/support_5_shots_joint", sep="\t", header=None)
	support_5shot_seen = pd.read_csv("./SMAN/SMAN_raw_data/NLUE/KFold_1/support_5_shots_seen", sep="\t", header=None)	
	novel_test_tsv = pd.read_csv("./SMAN/SMAN_raw_data/NLUE/KFold_1/test_novel.tsv", sep="\t", header=None)
	joint_test_tsv = pd.read_csv("./SMAN/SMAN_raw_data/NLUE/KFold_1/test_joint.tsv", sep="\t", header=None)
	train_tsv = pd.read_csv("./SMAN/SMAN_raw_data/NLUE/KFold_1/train_seen.tsv", sep="\t", header=None)

	support_1shot_novel.to_csv(inter_data_path + "support_1_shots_novel", sep="\t", header=None, index=False)
	support_1shot_seen.to_csv(inter_data_path + "support_1_shots_seen", sep="\t", header=None, index=False)
	support_1shot_joint.to_csv(inter_data_path + "support_1_shots_joint", sep="\t", header=None, index=False)
	support_5shot_novel.to_csv(inter_data_path + "support_5_shots_novel", sep="\t", header=None, index=False)
	support_5shot_seen.to_csv(inter_data_path + "support_5_shots_seen", sep="\t", header=None, index=False)
	support_5shot_joint.to_csv(inter_data_path + "support_5_shots_joint", sep="\t", header=None, index=False)
	train_tsv.to_csv(inter_data_path + "train_seen.tsv", sep="\t", header=None, index=False)
	novel_test_tsv.to_csv(inter_data_path + "test_novel.tsv", sep="\t", header=None, index=False)
	joint_test_tsv.to_csv(inter_data_path + "test_joint.tsv", sep="\t", header=None, index=False)
	return




def custom_data_helper(fsl_labels, seen_labels, label_to_utt, inter_data_path, num_shots):
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
		size_train = int(len(imp_utts)/2)
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
		size_train = int(len(imp_utts)/2)
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

	train_data.to_csv(inter_data_path + "train_seen.tsv", sep="\t", index=False, header=False)
	test_data_novel.to_csv(inter_data_path + "test_novel.tsv", sep="\t", index=False, header=False)
	test_data_joint.to_csv(inter_data_path + "test_joint.tsv", sep="\t", index=False, header=False)
	support_novel.to_csv(inter_data_path + "support_" + str(num_shots) + "_shots_novel", sep="\t", index=False, header=False)
	support_seen.to_csv(inter_data_path + "support_" + str(num_shots) + "_shots_seen", sep="\t", index=False, header=False)
	support_joint.to_csv(inter_data_path + "support_" + str(num_shots) + "_shots_joint", sep="\t", index=False, header=False)
	return





def helper(seen_labels, fsl_labels, label_to_utt):
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

	train_data.to_csv(inter_data_path + "train_seen.tsv", sep="\t", index=False, header=False)
	test_data_novel.to_csv(inter_data_path + "test_novel.tsv", sep="\t", index=False, header=False)
	test_data_joint.to_csv(inter_data_path + "test_joint.tsv", sep="\t", index=False, header=False)
	support1_novel.to_csv(inter_data_path + "support_1_shots_novel", sep="\t", index=False, header=False)
	support1_seen.to_csv(inter_data_path + "support_1_shots_seen", sep="\t", index=False, header=False)
	support1_joint.to_csv(inter_data_path + "support_1_shots_joint", sep="\t", index=False, header=False)
	support5_novel.to_csv(inter_data_path + "support_5_shots_novel", sep="\t", index=False, header=False)
	support5_seen.to_csv(inter_data_path + "support_5_shots_seen", sep="\t", index=False, header=False)
	support5_joint.to_csv(inter_data_path + "support_5_shots_joint", sep="\t", index=False, header=False)


def process_CLINC150(inter_data_path, num_shots):
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

	helper(seen_labels, fsl_labels, label_to_utt)
	return







def process_BANKING77(inter_data_path, num_shots):
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

	helper(seen_labels, fsl_labels, label_to_utt)
	return



def process_ATIS(inter_data_path, num_shots):
	clean_cache()
	Repo.clone_from("https://github.com/howl-anderson/ATIS_dataset.git", "./cache/")
	train_path = "./cache/data/standard_format/rasa/train.json"
	test_path = "./cache/data/standard_format/rasa/test.json"
	f = open(train_path, "r")
	data_train = json.load(f)
	f.close()
	f = open(test_path, "r")
	data_test = json.load(f)
	f.close()

	labels = []
	utts = []

	for i in range(len(data_test['rasa_nlu_data']['common_examples'])):
		utts.append(data_test['rasa_nlu_data']['common_examples'][i]['text'])
		labels.append(data_test['rasa_nlu_data']['common_examples'][i]['intent'])

	for i in range(len(data_train['rasa_nlu_data']['common_examples'])):
		utts.append(data_train['rasa_nlu_data']['common_examples'][i]['text'])
		labels.append(data_train['rasa_nlu_data']['common_examples'][i]['intent'])

	labels = [l.replace("+", " ") for l in labels]
	labels = [l.replace("_", " ") for l in labels]

	utt_to_label = {}
	for i,utt in enumerate(utts):
		utt_to_label[utt] = labels[i]

	all_labels = list(np.unique(labels))

	label_to_utt = {}
	for l in all_labels:
		label_to_utt[l] = []

	for key in label_to_utt.keys():
		for utt in utt_to_label.keys():
			if utt_to_label[utt] == key:
				label_to_utt[key].append(utt)

	for l in all_labels:
		if len(label_to_utt[l]) < 10:
			del label_to_utt[l]

	all_labels = list(np.unique(list(label_to_utt.keys())))
	random.seed(30)
	fsl_labels = random.sample(all_labels, 4)
	seen_labels = []
	for l in all_labels:
		if l not in fsl_labels:
			seen_labels.append(l)

	helper(seen_labels, fsl_labels, label_to_utt)
	return

def process_CUSTOM_DATA(inter_data_path, num_shots, config, path="./demo_data.csv"):
	data = pd.read_csv(path)
	utts = data["1"].to_list()
	labels = data["0"].to_list()

	utt_to_label = {}
	for i,utt in enumerate(utts):
		utt_to_label[utt] = labels[i]

	all_labels = list(np.unique(labels))
	random.seed(30)
	fsl_labels = random.sample(all_labels, int(((config["percent_novel_classes"]/100))*len(all_labels)))
	print("Novel class labels:")
	print(fsl_labels)
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

	custom_data_helper(fsl_labels, seen_labels, label_to_utt, inter_data_path, num_shots)

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

	if dataset_type == 'SNIPS':
		process_SNIPS(inter_data_path, num_shots)
	elif dataset_type == 'NLUE':
		process_NLUE(inter_data_path, num_shots)
	elif dataset_type == 'CLINC150':
		process_CLINC150(inter_data_path, num_shots)
	elif dataset_type == 'BANKING77':
		process_BANKING77(inter_data_path, num_shots)
	elif dataset_type == 'ATIS':
		process_ATIS(inter_data_path, num_shots)

	else:
		process_CUSTOM_DATA(inter_data_path, num_shots, config)