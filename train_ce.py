import math
import os.path
from utils import mkdir_p
import datasets
import sys
import torch
from sentence_transformers import InputExample, CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from torch.utils.data import DataLoader
from tqdm import tqdm

if len(sys.argv)>3:
    data=sys.argv[1] # TREC or CLEF
    dataset_path=sys.argv[2] # Path of the coped folder
    top_sens=sys.argv[3]
    passage_size = int(top_sens * 100)
    json_required = True
else:
    data = 'TREC'
    dataset_path='/home/ubuntu/'
    top_sens=0.20
    passage_size=int(top_sens*100)
    json_required = True


dataset = datasets.load_dataset("json", data_files={"train": [f'''train_qid_{data}_bm25_pass_{passage_size}.jsonl''']})


train_samples = []
for row in tqdm(dataset['train']):
    train_samples.append(InputExample(
        texts=[row['query'], row['top_sentences'].replace("\t"," ")], label=float(row['label'])
    ))

dataset = datasets.load_dataset("json", data_files={"test": [f'''test_qid_{data}_bm25_pass_{passage_size}.jsonl''']})

dataset_pos = dataset.filter(
    lambda x: True if x['label'] == 0 else False
)

dataset_neg = dataset.filter(
    lambda x: True if x['label'] == 1 else False
)

dev_sample={}
for i in dataset_pos['test']:
    if i['qid'] not in dev_sample:
        dev_sample[i['qid']]={
            'query': i['query'],
            'positive': set(),
            'negative':set()
        }
    if i['qid'] in dev_sample:
        dev_sample[i['qid']]['positive'].add(i['top_sentences'].replace("\t"," "))
    for j in dataset_neg['test']:
        if j['qid']==i['qid']:
            dev_sample[i['qid']]['negative'].add(j['top_sentences'])



torch.manual_seed(47)
model_name = 'biobert-v1.1'

best_score=0
model_path = f'''./cross_encoder_MRR_{model_name.split("/")[-1]}'''
result_folder = f'''{model_path}/result'''
mkdir_p(result_folder)

result_file = f'''{model_path}/result/no_c_score.csv'''

for batch_number, batch in enumerate([2,3,4,6,8,10,12]):
    for epoch_number, epoch in enumerate([1,2,3,4,5,6,7,8,9]):
        print(batch, epoch)
        train_batch_size = batch

        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
        evaluator = CERerankingEvaluator(dev_sample, name='train-eval')
        model_save_path = f'''./cross_encoder_MRR_{model_name.split("/")[-1]}_{data}_20/cross_encoder_{epoch}_{train_batch_size}_passage'''
        mkdir_p(model_save_path)

        if not os.path.isfile(model_save_path + "/pytorch_model.bin"):
            print("Training")
            warmup_steps = math.ceil(len(train_dataloader) * epoch * 0.1)  # 10% of train data for warm-up

            model = CrossEncoder(model_name, num_labels=1, max_length=512,
                                 automodel_args={'ignore_mismatched_sizes': True})

            # Train the model
            model.fit(train_dataloader=train_dataloader,
                      evaluator=evaluator,
                      epochs=epoch,
                      evaluation_steps=2000,
                      warmup_steps=warmup_steps,
                      output_path=model_save_path,
                      use_amp=True,
                      )
