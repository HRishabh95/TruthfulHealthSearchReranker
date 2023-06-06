from nltk import tokenize
from scipy import spatial
import torch
import math
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import language_tool_python

tool=language_tool_python.LanguageTool('en-US')

def cal_grammar_score(sentence):
    import numpy as np
    matches = tool.check(sentence)
    count_errors = len(matches)
    scores_word_based_sentence=count_errors
    word_count = len(sentence.split())
    sum_count_errors_word_based = np.sum(scores_word_based_sentence)
    score_word_based = 1 - (sum_count_errors_word_based / word_count)

    return np.mean(score_word_based)


from happytransformer import HappyTextToText, TTSettings
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
args = TTSettings(num_beams=5, min_length=1)
model_name = "pritamdeka/S-BioBert-snli-multinli-stsb"
model = SentenceTransformer(model_name)



def square_rooted(x):
    return math.sqrt(sum([a*a for a in x]))

def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return numerator/float(denominator)

# first_stage_rank=pd.read_csv('/home/ubuntu/rupadhyay/CREDPASS/Clef2020_BM25_clean_100.csv',sep='\t')
first_stage_rank=pd.read_csv('/home/ubuntu/rupadhyay/CREDPASS/TREC2020_BM25_clean_100.csv',sep='\t')

top_10_sents=[]
for ii,rows in first_stage_rank.iterrows():
    print(ii)
    tmp_list=[]
    vect=[]
    try:
        query_vec=model.encode([rows['query']],show_progress_bar=False)[0]
        sbuf=tokenize.sent_tokenize(rows['text'])
        for kk,text in enumerate(sbuf):
            text=happy_tt.generate_text(text).text
            if 4<len(text.split())<80:

                vect.append([text, model.encode([text],show_progress_bar=False)])
        simil=[]
        for jj,vec in enumerate(vect):

            simil.append([vec[0],1 - spatial.distance.cosine(query_vec, vec[1][0])])

        d=sorted(simil,
               key=lambda x: -x[1])
        top=int(0.25*len(simil))
        top_10_d=d[:top]
        correct_top=[]
        for i in top_10_d:
            sco=cal_grammar_score(i[0])
            if sco>=0.8:
                correct_top.append([i[0],i[1]])

        flat_list_text="\t".join([sublist[0] for sublist in correct_top if sublist[1] > 0.1])
        flat_list_score=",".join([str(sublist[1]) for sublist in correct_top if sublist[1] > 0.1])

        top_10_sents.append([
            rows['qid'],
            rows['docid'],
            rows['docno'],
            rows['rank'],
            rows['score'],
            rows['query'],
            rows['text'],
            flat_list_text,
            flat_list_score])
    except:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

top_10_sents_df=pd.DataFrame(top_10_sents)
top_10_sents_df.columns=['qid','docid','docno','rank','score','query','text','top_sentences','top_scores']
top_10_sents_df.to_csv('/home/ubuntu/rupadhyay/CREDPASS/trec2020_BM25_biobert_nltk_correct_sent_25.csv',sep='\t',index=False)