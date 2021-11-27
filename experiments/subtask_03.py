import json
import pandas as pd
import spacy
from sklearn.cluster import DBSCAN
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
from sense2vec import Sense2VecComponent
pd.set_option("max_colwidth", None)

# subtask-3
path_to_load = '../../03_dataset/task_01/subtask3-coreference/en-train.json'
sens_to_vec_path = 'E:/Datasets/sense2vec/s2v_old'

def read(path):
    """
    Reads the file from the given path (json file).
    Returns list of instance dictionaries.
    """
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for instance in file:
            data.append(json.loads(instance))

    return data

def convert_to_scorch_format(docs, cluster_key="event_clusters"):
    # Merge all JSON documents' clusters in a single list

    all_clusters = []
    for idx, doc in enumerate(docs):
        for cluster in doc[cluster_key]:
            all_clusters.append([str(idx) + "_" + str(sent_id) for sent_id in cluster])

    all_links = sum([list(itertools.combinations(cluster,2)) for cluster in all_clusters],[])
    all_events = [event for pair in all_links for event in pair]

    return all_links, all_events


data_json = read(path_to_load)
data_df = pd.DataFrame(data_json)
print(data_df.dtypes)
print(data_df.head())
print(data_df.columns)

all_links, all_events = convert_to_scorch_format(data_json)


nlp = spacy.load('en_core_web_md') #'en_core_web_trf' 'en_core_web_md'
# spacy.load("en_core_web_trf", disable=['transformer', 'tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
# nlp.disable_pipes('transformer', 'tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer')
# nlp.disable_pipes('tok2vec')
# nlp.add_pipe('tok2vec')
#nlp.add_pipe("sentencizer")
# nlp.initialize()

# init sense2vec
s2v = nlp.add_pipe("sense2vec")
s2v.from_disk(sens_to_vec_path)
print(f'NLP pipes: {nlp.pipe_names}')

docs = []
sent_vecs = {}

for idx, sent in enumerate(data_df.sentences):
    for i, sent_p in enumerate(sent):
        print(idx, i) #, sent_p)
        doc = nlp(sent_p)
        docs.append(doc)
        sent_vecs.update({sent_p: doc.vector})

sentences = list(sent_vecs.keys())
vectors = list(sent_vecs.values())

x = np.array(vectors)
n_classes = {}

# select best epsilon
for eps in tqdm(np.arange(0.01, 0.4, 0.005)):
    dbscan = DBSCAN(eps=eps, min_samples=2, metric='cosine').fit(x)
    n_classes.update({eps: len(pd.Series(dbscan.labels_).value_counts())})

print(n_classes)
plt.plot(n_classes.keys(), n_classes.values())

dbscan = DBSCAN(eps=0.0454, min_samples=2, metric='cosine').fit(x) # sm_w_s2t: 0.1459
results = pd.DataFrame({'label': dbscan.labels_, 'sent':sentences})

print('Class balance:')
print(results.groupby(['label']).size())
plt.figure()
results['label'].hist(bins=20)

examples = results['sent'][results.label==1]
examples

plt.show()
print()

def spacy_tokenize(data):
    # remove stop words
    # stop_words = spacy.lang.en.stop_words.STOP_WORDS
    print('Tokenize doc...')

    for i,text in enumerate(data):
        doc = nlp(text['sentences'])
        tokenized = [(token.text) for token in doc]   # best accuracy
        docs['text_tokenized'] = tokenized
        # lemmatized = [(token.lemma_) for token in doc]
        # text['text_lemmatized'] = lemmatized
        # no_stop_words = [token for token in lemmatized if token not in stop_words]
        # text['text_no_stop'] = no_stop_words
        print(f'Processing: {i}')