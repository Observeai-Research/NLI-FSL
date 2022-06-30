# Exploring the Limits of Natural Language Inference Based Setup for Few-Shot Intent Detection
This repository provides the implementation of the models used in the NLI based few shot learning. Refer to the [arxiv](https://arxiv.org/pdf/2112.07434.pdf) pre-print for more information.

## Setup
Use Python version ```3.7.1``` to run all the experiments and data creation. To install all the required dependencies to run all the experiments use the command ```pip install -r requrements.txt```.

## Usage
### Step 1: Making Few Shot Splits (Intermediate Data)
Please use the file ```datamaking.py``` to re-construct the dataset splits that we evaluated the models upon. 

#### Config:
* ```--dataset```: Choose any of the datasets we experimented with, "SNIPS", "NLUE", "ATIS", "CLINC150" or "BANKING77". This will re-construct our 1-shot and 5-shot Few Shot Learning and Generalized Few Shot Learning splits (intermediate dataset). To make the Few Shot Split  for your own custom dataset, use any other name. Refer to the format of ```demo_data.csv```.
* ```--inter_data_path```: The path where the intermediate dataset will be stored
* ```--percent_novel_classes```: Percentage of novel classes you want in the intermediate few shot dataset. 
* ```--num_samples_per_class```: Denotes the number of shots (only required in case of custom dataset. By default, we make 1-shot and 5-shot splits of the other datasets)

#### Example:
```
python datamaking.py --dataset="SNIPS" --inter_data_path="./dummy_data/" --percent_novel_classes=33
```

### Step 2: Training model
Please use the file ```nli-fsl-training.py``` to train the NLI-FSL method over the intermediate dataset obtained in the previous step. or use ```proto-train.py``` to train the ProtoBERT baseline.

#### Config:
* ```--inter_data_path```: The path where the intermediate dataset constructed in the previous step is stored.
* ```--device```: Device that you want to train the model upon (default='cuda')
* ```--num_samples_per_class```: Number of shots in the support dataset. (use 0 for zero shot learning on NLI-FSL).
* ```--seen_class_data```: If you want to train the model for NLI-FSL with seen class data as well, set it to be "yes" (default="yes")
* ```--ratio```: (ONLY FOR NLI-FSL) The ratio between the number of non-entailed pairs to be sampled and the number of entailed pairs. (default=2)
* ```--save_folder_name```: Path of the folder where you want to store the fine-tuned BERT model.
* ```--batch_size```: Batch size used while fine-tuning the model (default=64)
* ```--max_seq_len```: Maximum sequence length in BERT input tokens (default=64)
* ```--epochs```: Number of epochs to fine-tune the BERT model (default=3)
* ```--learning_rate```: Learning rate of the model while learning (default=2e-6)
#### Example NLI-FSL:
```
python nli-fsl-training.py --inter_data_path="./dummy_data/" --num_samples_per_class=5 --save_folder_name="./BERT/" --ratio=2 --epochs=3 --seen_class_data="yes"
```

#### Example ProtoBERT:
```
python proto-train.py --inter_data_path="./dummy_data/" --num_samples_per_class=5 --save_folder_name="./BERT/"
```

### Step 3: Inference
Please use the file ```nli-fsl-inference.py``` to infer the NLI-FSL method or use ```proto-infer.py``` to infer the ProtoBERT over the query data in the intermediate dataset obtained in Step 1. 

#### Config:
* ```--inter_data_path```: The path where the intermediate dataset constructed in the previous step is stored.
* ```--device```: Device that you want to train the model upon (default='cuda')
* ```--num_samples_per_class```: Number of shots in the support dataset.
* ```--generalized```: Whether you want to perform Few shot or Generalized few shot learning (default="no")
* ```--model_path```: Path of the folder where the model checkpoint from the previous step is stored.
* ```--batch_size```: Batch size used while fine-tuning the model (default=64)
* ```--max_seq_len```: Maximum sequence length in BERT input tokens (default=64)
* ```--use_saved_model```: For ProtoBERT only if you want to infer in the absence of seen class data (default="yes")

#### Example NLI-FSL:
```
python nli-fsl-inference.py --inter_data_path="./dummy_data/" --num_samples_per_class=5 --model_path="./BERT/" --generalized="no"
```

#### Example ProtoBERT:
```
python proto-infer.py --inter_data_path="./dummy_data/" --num_samples_per_class=5 --model_path="./BERT/" --generalized="yes"
```

## Other Experiments:
### Incremental Label Space
Please use the file ```incremental_label_space.py``` to re-construct the dataset splits (for CLINC150 and BANKING77)that we evaluated the models upon. Note that the splits are created in the specified folder (similar to step 1) except, that the directory will consist of 10 folder from 10_percent to 100_percent of the novel classes used. The config for the file is as follows:

#### Config:
* ```--dataset```: The name of dataset you want to re-construct the splits for (Use "CLINC150" or "BANKING77")
* ```--inter_data_path```: The directory where you want the 10 folders to be created.

#### Example:
```
python incremental_label_space.py --dataset="CLINC150" --inter_data_path="./dummy_data/"
```

### Incremental seen class data
Please use the file ```incremental_seen_class.py``` to re-construct the dataset splits (for CLINC150 and BANKING77) that we evaluated the models upon. Note that the splits are created in the specified folder (similar to step 1) except, that the directory will consist of 5 folders from 20_percent to 100_percent of the base/seen class dataset used. The config for the file is as follows:

#### Config:
* ```--dataset```: The name of dataset you want to re-construct the splits for (Use "CLINC150" or "BANKING77")
* ```--inter_data_path```: The directory where you want the 5 folders to be created.

#### Example:
```
python incremental_seen_class.py --dataset="CLINC150" --inter_data_path="./dummy_data/"
```


### Incremental number of shots
Please use the file ```incremental_shots.py``` to re-construct the dataset splits (for CLINC150) that we evaluated the models upon. Note that the splits are created in the specified folder (similar to step 1). The config for the file is as follows:

#### Config:
* ```--num_samples_per_class_list```: Provide the list of shots you want, example '1 5 10'. By default it creates the same number of shot splits as our experiments ('1 5 10 20').
* ```--inter_data_path```: The directory where you want the 10 folders to be created.

#### Example:
```
python incremental_shots.py --inter_data_path="./dummy_data/" --num_samples_per_class_list='1 5 10 20 30'
```

## SMAN
Change the path to the directory, ```./SMAN/code/``` and then run the file ```main.py``` to train and obtain the results on the intermediate dataset.
#### Config:
* ```--inter_data_path```: The path where the intermediate dataset constructed in the previous step is stored.
* ```--device```: Device that you want to train the model upon (default='cuda')
* ```--num_samples_per_class```: Number of shots in the support dataset.
* ```--fasttext_path```: Path to the directory, where there are the fasttext embeddings. Use the [link](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip) to download the fasttext embeddings and put them in the folder ```./SMAN/fasttext/```
* ```--tgt```: Generalized zero shot learning or normal zero shot learning, choose "joint" for generalized FSL.
* ```--ckpt_dir```: Saved directory for checkpoint.
* ```--num_run```: number of runs for the SMAN method.



### Example:
```
python main.py --inter_data_path='../../dummy_data/' --tgt='joint' --num_samples_per_class=5
```

## MLMAN
Change the path to the directory, ```./MLMAN/code/``` and then run the file ```main.py``` to train and obtain the results on the intermediate dataset.
#### Config:
* ```--inter_data_path```: The path where the intermediate dataset constructed in the previous step is stored.
* ```--device```: Device that you want to train the model upon (default='cuda')
* ```--num_samples_per_class```: Number of shots in the support dataset.
* ```--glove_path```: Path to the directory, where there are the glove 50 dimensional embeddings. Use the [link](https://cloud.tsinghua.edu.cn/f/b14bf0d3c9e04ead9c0a/?dl=1) to download the fasttext embeddings and put them in the folder ```./MLMAN/glove/```
* ```--tgt```: Generalized zero shot learning or normal zero shot learning, choose "joint" for generalized FSL.
* ```--ckpt_dir```: Saved directory for checkpoint.
* ```--num_run```: number of runs for the MLMAN method.

Base code was taken from MLMAN ([link](https://github.com/ZhixiuYe/MLMAN))

### Example:
```
python main.py --inter_data_path='../../dummy_data/' --tgt='joint' --num_samples_per_class=5
```

## HATT
Change the path to the directory, ```./HATT/code/``` and then run the file ```main.py``` to train and obtain the results on the intermediate dataset.
#### Config:
* ```--inter_data_path```: The path where the intermediate dataset constructed in the previous step is stored.
* ```--device```: Device that you want to train the model upon (default='cuda')
* ```--num_samples_per_class```: Number of shots in the support dataset.
* ```--glove_path```: Path to the directory, where there are the glove 50 dimensional embeddings. Use the [link](https://cloud.tsinghua.edu.cn/f/b14bf0d3c9e04ead9c0a/?dl=1) to download the fasttext embeddings and put them in the folder ```./MLMAN/glove/```
* ```--tgt```: Generalized zero shot learning or normal zero shot learning, choose "joint" for generalized FSL.
* ```--ckpt_dir```: Saved directory for checkpoint.
* ```--num_run```: number of runs for the HATT method.

Base code was taken from HATT-Proto ([link](https://github.com/thunlp/HATT-Proto))

### Example:
```
python main.py --inter_data_path='../../dummy_data/' --tgt='joint' --num_samples_per_class=5
```


## Zero-Shot DDN Usage
#### Config:
* ```--inter_data_path```: The path where the intermediate dataset constructed in the previous step is stored.
* ```--device```: Device that you want to train the model upon (default='cuda')
* ```--generalized```: Generalized zero shot learning or normal zero shot learning (default="no")
* ```--encoder```: Encoder can be chosen from "USE" (Universal Sentence Encoder) or "SBERT" (SentenceBERT).

### Example:
```
python ZS-DDN.py --inter_data_path='./dummy_data/' --num_samples_per_class=5 --generalized="no" --encoder="SBERT"
```