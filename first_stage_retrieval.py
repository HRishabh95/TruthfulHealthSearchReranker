import os
import pandas as pd
import pyterrier as pt
import sys
import re
import string
from utils import *
# Set JAVA_HOME environment variable
os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-1.11.0-openjdk-amd64/"

# Check if PyTerrier is already started
if not pt.started():
    pt.init()

# Constants
PUNCTUATIONS = string.punctuation.replace(".", "")

def remove_punctuation(text):
    trans = str.maketrans(dict.fromkeys(PUNCTUATIONS, ' '))
    return text.translate(trans)

def remove_whitespaces(text):
    return " ".join(text.split())

def clean_en_text(text):
    text = re.sub(r"[^A-Za-z0-9(),.!?%\'`]", " ", text)
    text = re.sub(r'\d+', '', text)
    text = remove_punctuation(text)
    text = remove_whitespaces(text)
    return text.strip().lower()



if len(sys.argv)>3:
    data_set=sys.argv[1] # TREC or CLEF
    indexing=sys.argv[2] # True or False
    dataset_path=sys.argv[3] # Path of the coped folder
else:
    data_set = 'TREC'
    indexing = False
    dataset_path='/home/ubuntu/'

config={'TREC':{'folder_path':f'''{dataset_path}/TREC''',
                'file_path':f'''{dataset_path}/TREC/TREC2020_1M_labeled.csv''',
                'index_path':f'''{dataset_path}/TREC/trec2020_bm25''',
                'topics':f'''{dataset_path}/TREC/topics.csv''',
                'result_name':f'''{dataset_path}/TREC/trec2020_BM25.csv''',
                'qrels':f'''{dataset_path}/TREC/trec_qrels_top.csv''',
                'final_retrieved_name':f'''{dataset_path}/TREC/TREC2020_BM25_clean_100.csv'''},
        'CLEF':{'folder_path':f'''{dataset_path}/CLEF''',
                'file_path':f'''{dataset_path}/CLEF/Clef2020_1M_labeled.csv''',
                'index_path':f'''{dataset_path}/CLEF/clef2020_bm25''',
                'topics':f'''{dataset_path}/CLEF/clef_topics.csv''',
                'result_name': f'''{dataset_path}/CLEF/clef2020_BM25.csv''',
                'qrels': f'''{dataset_path}/CLEF/clef_qrels_top.csv''',
                'final_retrieved_name': f'''{dataset_path}/CLEF/CLEF2020_BM25_clean_100.csv'''}}

if indexing:
    index_doc=pd.read_csv(config[data_set]['file_path'],sep='\t')
    index_doc=index_doc.dropna(subset=['text'])
    index_doc.drop_duplicates(subset=['text'],inplace=True)

    index_path=config[data_set]['index_path']
    index_doc=index_doc[['docno','text']]
    if not os.path.exists(f'''{index_path}/data.properties'''):
        indexer = pt.DFIndexer(index_path, overwrite=True, verbose=True, Threads=8)
        indexer.setProperty("termpipelines", "PorterStemmer") # Removes the default PorterStemmer (English)
        indexref3 = indexer.index(index_doc["text"], index_doc)
    else:
        indexref3 = pt.IndexRef.of(f'''{index_path}/data.properties''')

indexref3 = pt.IndexRef.of(f'''{config[data_set]['index_path']}/data.properties''')
BM251 = pt.BatchRetrieve(indexref3, num_results=500, controls = {"wmodel": "BM25"})
topics=pt.io.read_topics(config[data_set]['topics'],format='singleline')
results=BM251.transform(topics)
results = results[~results["qid"].isin(['28'])]
results['Q0']=0
result=results[['qid','Q0','docno','rank','score']]
results.to_csv(f'''{config[data_set]['result_name']}''',sep=' ', index=None, header=None)


qrels_path=f'''{config[data_set]['qrels']}'''
qrels = pt.io.read_qrels(qrels_path)
eval = pt.Utils.evaluate(results,qrels,metrics=["ndcg"], perquery=True)
print(eval)

docs=pd.read_csv(config[data_set]['file_path'],sep='\t')
docs=docs[['docno','text']]
merged_results=pd.merge(results,docs,on=['docno'])
merged_results.to_csv(f'''{config[data_set]['final_retrieved_name']}''',sep='\t',index=False)