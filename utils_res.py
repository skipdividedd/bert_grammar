import os
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go



class Init:

    def __init__(self, path, lang, d_name):

        with open(f'{path}scores_{lang}_{d_name}.pkl', 'rb') as f:
            self.scores = pickle.load(f)
    
        with open(f'{path}scores_c_{lang}_{d_name}.pkl', 'rb') as f:
            self.scores_c = pickle.load(f)
            
        # нейроны.......
        with open(f'{path}neurons_{lang}_{d_name}.pkl', 'rb') as f: # все с отсечками
            self.ordered_neurons = pickle.load(f)
            
        with open(f'{path}top_n_{lang}_{d_name}.pkl', 'rb') as f: #тут 10 проц
            self.top_neurons = pickle.load(f)
            
        with open(f'{path}bottom_n_{lang}_{d_name}.pkl', 'rb') as f: #тут 
            self.bottom_neurons = pickle.load(f)
            
            
        with open(f'{path}threshold_{lang}_{d_name}.pkl', 'rb') as f: # с "трешхолдом"
            self.ordered_neurons_thres = pickle.load(f)
            
        # тут с аблейшн.....
        with open(f'{path}scores_keep_bot_{lang}_{d_name}.pkl', 'rb') as f:
            self.scores_keep_bot = pickle.load(f)
            
        with open(f'{path}scores_keep_top_{lang}_{d_name}.pkl', 'rb') as f:
            self.scores_keep_top = pickle.load(f)
            
        with open(f'{path}scores_keep_thres_{lang}_{d_name}.pkl', 'rb') as f:
            self.scores_keep_thres = pickle.load(f)
            
        # для контрол таск 
        with open(f'{path}scores_keep_top_c_{lang}_{d_name}.pkl', 'rb') as f:
            self.scores_keep_top_c = pickle.load(f)
            
        with open(f'{path}scores_keep_thres_c_{lang}_{d_name}.pkl', 'rb') as f:
            self.scores_keep_thres_c = pickle.load(f)
            
        # сколько данных
        with open(f'{path}size_{lang}_{d_name}.pkl', 'rb') as f:
            self.size = pickle.load(f)


def common_neurons(d1, d2):
    d1 = dict(sorted(d1.items())) 
    d2 = dict(sorted(d2.items())) 
    common_cats = []
    for k1 in d1.keys():
        for k2 in d2.keys():   
            if k1 == k2:
                common_cats.append(k1)
            
    df = pd.DataFrame(columns=common_cats)
    df = df.fillna(0)
    for cat in common_cats:
        common_neurons = []
        p = set(d1[cat]) & set(d2[cat])
        common_neurons.append(len(p))
        df[cat] = common_neurons
    return df   


def common_neurons_percentage(d1, d2):
    d1 = dict(sorted(d1.items())) 
    d2 = dict(sorted(d2.items())) 
    common_cats = []
    for k1 in d1.keys():
        for k2 in d2.keys():   
            if k1 == k2:
                common_cats.append(k1)
            
    df = pd.DataFrame(columns=common_cats)
    df = df.fillna(0)
    for cat in common_cats:
        common_neurons = []
        p = len(set(d1[cat]) & set(d2[cat])) * 100 / len((set(d1[cat]) | set(d2[cat])))
        common_neurons.append(round(p, 2))
        df[cat] = common_neurons
    return df 


def common_neurons_percentage_multiple(dct_acc1, dct_acc2, le=[5, 10, 15, 20, 25, 30]):
    cats = ['ADJ_Gender', 'NOUN_Number', 'NOUN_Case', 'VERB_Aspect', 'VERB_Person', 'VERB_Tense']
    d1 = {}
    d2 = {}
    for c in cats:
        d1[c] = dct_acc1[c]
        d2[c]  = dct_acc2[c] 
    common_cats = []
    for k1 in d1.keys():
        for k2 in d2.keys():   
            if k1 == k2:
                common_cats.append(k1)
    le = sorted(le, reverse=True)
    le_id = [str(i) + '%' for i in le]        
    df = pd.DataFrame(index=le_id, columns=common_cats)
    df = df.fillna(0)

    for cat in common_cats:
        common_neurons = []
        for l in le:
            p = len(set(d1[cat][0][:d1[cat][1][l-1]]) & set(d2[cat][0][:d2[cat][1][l-1]])) * 100 / len((set(d1[cat][0][:d1[cat][1][l-1]]) | set(d2[cat][0][:d2[cat][1][l-1]])))
            common_neurons.append(round(p, 2))
        df[cat] = common_neurons
    return df 


def common_neurons_multiple(d1, d2, le=[5, 10, 20, 30, 40, 50, 75, 80, 90, 99]):
    d1 = dict(sorted(d1.items())) 
    d2 = dict(sorted(d2.items())) 
    common_cats = []
    for k1 in d1.keys():
        for k2 in d2.keys():   
            if k1 == k2:
                common_cats.append(k1)
    le_id = [str(i) + '%' for i in le]        
    df = pd.DataFrame(index=le_id, columns=common_cats)
    df = df.fillna(0)

    for cat in common_cats:
        common_neurons = []
        for l in le:
            p = set(d1[cat][0][:d1[cat][1][l-1]]) & set(d2[cat][0][:d2[cat][1][l-1]])
            common_neurons.append(len(p))
        df[cat] = common_neurons
    return df 


def common_heatmap(d):
    d = dict(sorted(d.items()))
    d = {k:[v] for k, v in d.items()}
    cats = [k for k in d.keys()]
    
    df = pd.DataFrame(index=cats, columns=cats)
    df = df.fillna(0)

    for cat in cats:
        common_neurons = []
        for l in range(len(d.keys())):
            p = len(set(d[cat][0]) & set(d[cats[l]][0])) * 100 / len(set(d[cat][0]) | set(d[cats[l]][0]))
            common_neurons.append(f'{round(p, 2)}%')
        df[cat] = common_neurons
    return df 


def common_diff_heatmap(d1, d2):
    d1 = dict(sorted(d1.items())) 
    d1 = {k:[v] for k, v in d1.items()}
    d2 = dict(sorted(d2.items())) 
    d2 = {k:[v] for k, v in d2.items()}
    cats1 = [k for k in d1.keys()]
    cats2 = [k for k in d2.keys()]
    
    df = pd.DataFrame(index=cats1, columns=cats2)
    df = df.fillna(0)

    for cat2 in cats2:
        common_neurons = []
        for cat1 in cats1:
            p = len(set(d1[cat1][0]) & set(d2[cat2][0])) * 100 / len(set(d1[cat1][0]) | set(d2[cat2][0]))
            common_neurons.append(round(p, 2))
        df[cat2] = common_neurons
    return df