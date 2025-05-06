from __future__ import annotations

from nltk.corpus import brown, stopwords, wordnet as wn
from nltk.corpus.reader.wordnet import Synset
from nltk.stem import WordNetLemmatizer
from typing import Iterable

import math
import nltk
import random
import string
import time

class Sentence:
    '''
    This class expresses sentences, where a raw text form of sentence is given.
    '''

    stop = set(stopwords.words("english"))
    lemm = WordNetLemmatizer()
    synset_type = {'N': wn.NOUN, 'V': wn.VERB, 'J': wn.ADJ, 'R': wn.ADV, 'M': wn.VERB}
    
    def __init__(self, raw: str):
        self.tokens = nltk.word_tokenize(raw.lower())
        self.tokens = nltk.pos_tag(self.tokens)
        self.tokens = [(Sentence.lemm.lemmatize(w), p) for w, p in self.tokens
                       if w not in Sentence.stop and w not in string.punctuation]
        
        self.synsets: list[Synset] = []
        for w, p in self.tokens:
            if p[0] not in Sentence.synset_type:
                continue
            
            synsets = wn.synsets(w, pos=Sentence.synset_type[p[0]])
            if synsets:
                self.synsets.append(synsets[0])
            
    def similarity(self, sent: Sentence) -> float:
        if not self.synsets or not sent.synsets:
            return 0
        
        sim1 = [0] * len(self.synsets)
        sim2 = [0] * len(sent.synsets)
        
        for i, synset1 in enumerate(self.synsets):
            for j, synset2 in enumerate(sent.synsets):
                sim = synset1.wup_similarity(synset2)
                sim1[i] = max(sim1[i], sim)
                sim2[j] = max(sim2[j], sim)
        
        return (sum(sim1)/len(sim1) + sum(sim2)/len(sim2)) / 2

class Text:
    '''
    This class expresses text, which is a list of sentences.
    '''
    
    def __init__(self, sentences: Iterable[Sentence], sigma=2):
        self.sentences = sentences
        self.sigma = sigma
        
        self.sim_mat = self.eval_sim_mat()
        self.coeff_mat = self.eval_coeff_mat()
    
    def __len__(self):
        return len(self.sentences)
    
    def eval_sim_mat(self):
        '''
        This function evaluates similarity scores of every pairs of sentences.
        
        (TODO: replace sentence similarity into flow reliability.)
        '''
        
        N = len(self)
        mat = [[0]*N for _ in range(N)]
        
        for i1, sent1 in enumerate(self.sentences):
            for j, sent2 in enumerate(self.sentences[i1+1:]):
                i2 = i1 + 1 + j
                sim = sent1.similarity(sent2)
                
                mat[i1][i2] = sim
                mat[i2][i1] = sim
        
        return mat
    
    def eval_coeff_mat(self):
        '''
        If two sentences are located far away from each other,
        the contribution to cut score should be low.
        
        This function calculates those weights.
        '''
        
        N = len(self)
        mat = [[0]*N for _ in range(N)]
        
        for i in range(N):
            for j in range(N):
                x = i - j
                val = 1 / (math.sqrt(2 * math.pi) * self.sigma)
                val *= math.exp(-x**2 / (2 * self.sigma**2))
                mat[i][j] = val
        
        return mat
    
    def cut_score(self, k):
        '''
        Suppose the text is consist of N sequences,
        and we separate it into two parts: 1 to k and k+1 to N. (1 <= k < N)
        
        This function calculates the cut score between two parts, based on sentence similarity.
        
        (TODO: replace sentence similarity into flow reliability.)
        '''
        
        N = len(self)
        assert 1 <= k < N
        
        score_sum = 0
        coeff_sum = 0
        
        for i in range(k):
            for j in range(k, N):
                score_sum += self.coeff_mat[i][j] * self.sim_mat[i][j]
                coeff_sum += self.coeff_mat[i][j]
        
        return score_sum / coeff_sum

fileids = brown.fileids('news')
file_id = random.sample(fileids, 1)[0]

sentences = brown.sents(fileids=[file_id])
paragraphs = brown.paras(fileids=[file_id])

sents = [Sentence(' '.join(raw)) for raw in sentences[:20]]
sent_similarity = [[] for _ in range(len(sents)-1)]
cut_scores = [0] * (len(sents)-1)

text = Text(sents)

print([len(para) for para in paragraphs])
for k in range(1, len(text)):
    print(f'{k:2d}: {text.cut_score(k):.5f}')