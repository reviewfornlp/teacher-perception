import numpy as np


import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_svmlight_files

np.random.seed(1)
from collections import defaultdict
from copy import deepcopy
from scipy.stats import chisquare
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from collections import Counter
import pyconll
import random
from indictrans import Transliterator

def convertStringToset(data):
    feats = {}
    if data == "_":
        return {}
    for f in data.split("|"):
        k = f.split("=")[0]
        v = f.split("=")[1]
        feats[k]=v
    return feats

def find_agreement(feats1, feats2):
    shared = set()
    agreed = set()
    for feat in feats1:
        if feat in feats2:
            shared.add(feat)
            if feats1[feat] == feats2[feat]:
                agreed.add(feat)
    return shared, agreed

def get_vocab_from_set(input):
    word_to_id = {}
    if "NA" in input:
        word_to_id['NA'] = 0
    for i in input:
        if i == "NA":
            continue
        word_to_id[i] = len(word_to_id)
    id_to_word = {v:k for k,v in word_to_id.items()}
    return word_to_id, id_to_word

def transformRulesIntoReadable(feature, task, rel, relation_map, folder_name, pos_examples={}, source='eng'):
    global trn
    if source:
        trn = Transliterator(source=source, target='eng', build_lookup=True)
    else:
        trn = None

    task = task.lower()
    if task == 'wordorder':
        dependent = rel.split("-")[0]
        head = rel.split("-")[1]
        if rel in ['adjective-noun', 'numeral-noun']:
            head = 'nominal'
        if rel in ['noun-adposition']:
            dependent = 'nominal'
    elif task == 'agreement':
        if "-" in rel:
            dependent = rel.split("-")[1]
        else:
            dependent = "current word"
        head = 'head'
    elif task == "assignment":
        dependent = rel
        head = 'head'
    else:
        dependent = ''
        head = ''

    new_features = feature
    info = feature.split("_")

    def get_relation(info):
        lang_link = f'https://universaldependencies.org/'
        lang = folder_name.split('/')[3]
        lang_link = "https://autolex.co/"
        if info in relation_map:
            (value, link) = relation_map[info]
            #if not os.path.exists(f'{folder_name}/helper/{info.lower()}_relations.html'):
            #   lang_link = link

        elif info.lower() in relation_map:
            (value, link) = relation_map[info.lower()]
            #if not os.path.exists(f'{folder_name}/helper/{info.lower()}_relations.html'):
            #    lang_link = link
            #hovertext= f'<a href=" " title="for e.g. words like ..." style="background-color:#FFFFFF;color:#000000;text-decoration:none">{value}</a>'
            #value = hovertext
        elif info.split("@")[0] in relation_map or info.split("@")[0].lower() in relation_map:
            (value, link) = relation_map.get(info.split("@")[0].lower(), info.lower())
            #if not os.path.exists(f'{folder_name}/helper/{info.lower()}_relations.html'):
            #    lang_link = link
            #hovertext= f'<a href=" " title="for e.g. words like ..." style="background-color:#FFFFFF;color:#000000;text-decoration:none">{value}</a>'
            #value = hovertext
        else:
            value = info
        return value, lang_link

    if feature.startswith("spinehead"):
        return f'{head} is a word like= {info[1]}'

    elif feature.startswith('spine'):
        return f'{dependent} is a word like= {info[1]}'

    if len(info) == 2: #f'dep_{relation}_'
        if feature.startswith('wiki'):
            new_features = f'{dependent}\'s semantic class is= {info[1]}'
        elif feature == 'headmatch_True':
            new_features = f'the head agrees with its head on= {rel}'
        elif feature.startswith('lemma'):
            if trn:
                eng = trn.transform(info[1])
                new_features = f'{dependent} has lemma= {info[1]} ({eng})'
            else:
                new_features = f'{dependent} has lemma= {info[1]}'
        elif feature.startswith('headlemma'):
            if trn:
                eng = trn.transform(info[1])
                new_features = f'{dependent} is governed by= {info[1]} ({eng})'
            else:
                new_features = f'{dependent} is governed by= {info[1]}'
        elif feature.startswith('depdeplemma'):
            if trn:
                eng = trn.transform(info[1])
                new_features = f'{dependent} is governing= {info[1]} ({eng})'
            else:
                new_features = f'{dependent} is governing= {info[1]}'
        elif feature.startswith('headheadlemma'):
            if trn:
                eng = trn.transform(info[1])
                new_features = f'{head} is governed by= {info[1]} ({eng})'
            else:
                new_features = f'{head} is governed by= {info[1]}'
        elif feature.startswith('depheadlemma'):
            if trn:
                eng = trn.transform(info[1])
                new_features = f'{head} is also governing= {info[1]} ({eng})'
            else:
                new_features = f'{head} is also governing= {info[1]}'
        elif feature.startswith('neighborhood'):
            if trn:
                eng = trn.transform(info[1])
                new_features = f'{dependent} is nearby= {info[1]} ({eng})'
            else:
                new_features = f'{dependent} is nearby= {info[1]}'
        elif feature.startswith('left'):
            if trn:
                eng = trn.transform(info[1])
                new_features = f'before the {dependent} is= {info[1]} ({eng})'
            else:
                new_features = f'before the {dependent} is= {info[1]}'
        elif feature.startswith('right'):
            if trn:
                eng = trn.transform(info[1])
                new_features = f'after the {dependent} is= {info[1]} ({eng})'
            else:
                new_features = f'after the {dependent} is= {info[1]}'
        elif feature.startswith('lang'):
            new_features = f'lang is= {info[1]}'
        elif feature.startswith('srclem'):
            new_features = f'in English is= {info[1]}'
        elif feature.startswith('srcpos'):
            info_, link = get_relation(info[1])
            title = ''
            if len(pos_examples) > 0 and info[1] in pos_examples:
                title = f'For eg. words like {", ".join(pos_examples[info[1]])}'
            info[1] = f'<a href="{link}" title="{title}"> {info_} </a>'
            new_features = f'in English is= {info[1]}'
        else: #f'deppos_{pos}'

            info_, link = get_relation(info[1])

            if feature.startswith('deppos'):
                title = ''
                if len(pos_examples) > 0 and info[1] in pos_examples:
                    title = f'For eg. words like {", ".join(pos_examples[info[1]])}'
                info[1] = f'<a href="{link}" title="{title}"> {info_} </a>'
                new_features = f'{dependent} is a= {info[1]}'

            elif feature.startswith('headpos'):
                title = ''
                if len(pos_examples) > 0 and info[1] in pos_examples:
                    title = f'For eg. words like {", ".join(pos_examples[info[1]])}'
                info[1] = f'<a href="{link}" title="{title}"> {info_} </a>'
                new_features = f'{dependent} is governed by a= {info[1]}'

            elif feature.startswith('deprel'):
                info[1] = f'<a href="{link}" title=""> {info_} </a>'
                new_features = f'{dependent} is the= {info[1]}'

            elif feature.startswith('headrelrel'):
                info[1] = f'<a href="{link}" title=""> {info_} </a>'
                new_features= f'{dependent} is governed by= {info[1]}'

            elif feature.startswith('depdeppos'):
                title = ''
                if len(pos_examples) > 0 and info[1] in pos_examples:
                    title = f'For eg. words like {", ".join(pos_examples[info[1]])}'
                info[1] = f'<a href="{link}" title="{title}"> {info_} </a>'
                new_features = f'{dependent} is governing= {info[1]}'

            elif feature.startswith('depdeprel'):
                info[1] = f'<a href="{link}" title=""> {info_} </a>'
                new_features = f'{dependent} is governing= {info[1]}'

            elif feature.startswith('depheadrel') or feature.startswith('depheadpos'):
                title = ''
                if len(pos_examples) > 0 and info[1] in pos_examples:
                    title = f'For eg. words like {", ".join(pos_examples[info[1]])}'
                info[1] = f'<a href="{link}" title="{title}"> {info_} </a>'
                head_phrase = head
                if head_phrase == 'head':
                    new_features = f'{dependent} is nearby= {info[1]}'
                else:
                    new_features = f'{dependent} is nearby= {info[1]}'

            elif feature.startswith('svo'):
                new_features = f'anaphora= True'

            elif feature.startswith('agreepos'):
                title = ''
                if len(pos_examples) > 0 and info[1] in pos_examples:
                    title = f'For eg. words like {", ".join(pos_examples[info[1]])}'
                info[1] = f'<a href="{link}" title="{title}"> {info_} </a>'
                new_features = f'{dependent} agrees with its head= {info[1]}'

            elif feature.startswith('agreerel'):
                info[1] = f'<a href="{link}" title=""> {info_} </a>'
                new_features = f'{dependent} agrees with its head when {dependent} is a= {info[1]}'

            elif feature.startswith('agree'):
                title = ''
                if len(pos_examples) > 0 and info[1] in pos_examples:
                    title = f'For eg. words like {", ".join(pos_examples[info[1]])}'
                info[1] = f'<a href="{link}" title="{title}"> {info_} </a>'
                new_features = f'{dependent} agrees with its head when the head is a= {info[1]}'

    if len(info) == 3: #f'depposrel_{pos}_{relation}'
        info_, link = get_relation(info[-1])
        if feature.startswith("depposrel"):
            info_pos, pos_link = get_relation(info[1])
            title = ''
            if len(pos_examples) > 0 and info[1] in pos_examples:
                title = f'For eg. words like {", ".join(pos_examples[info[1]])}'
            info[1] = f'<a href="{pos_link}" title="{title}"> {info_pos} </a>'
            info[2] = f'<a href="{link}" title=""> {info_} </a>'
            new_features = f'{info[1]} is the dependent and is the= {info[2]}'

        elif feature.startswith('headrelrel'): #f'headrelrel_{head_pos}_{headrelation.lower()}'
            title = ''
            if len(pos_examples) > 0 and info[1] in pos_examples:
                title = f'For eg. words like {", ".join(pos_examples[info[1]])}'
            info[1] = f'<a href="{link}" title="{title}"> {info[1]} </a>'
            new_features = f'{dependent} has head-{info[1]} is a= {info_}'

        elif feature.startswith('headfeat'):
            feats_link = 'https://universaldependencies.org/u/feat/index.html'
            info[1] = f'<a href="{feats_link}" title=""> {info[1]} </a>'
            new_features = f'{dependent} is governed by a word with {info[1]}= {info[2]}'

        elif feature.startswith('depfeat'):
            feats_link = 'https://universaldependencies.org/u/feat/index.html'
            info[1] = f'<a href="{feats_link}" title=""> {info[1]} </a>'
            new_features = f'{dependent} with {info[1]}= {info[2]}'

        elif feature.startswith('wiki'):
            new_features = f'the head of the {dependent} has semantic class= {info[-1]}'

        elif feature.startswith("head") :#f'head_{head_pos}_{relation}'
            title = ''
            info_pos, pos_link = get_relation(info[1])
            if len(pos_examples) > 0 and info[1] in pos_examples:
                title = f'For eg. words like {", ".join(pos_examples[info[1]])}'
            info[1] = f'<a href="{pos_link}" title="{title}"> {info[1]} </a>'

            info[-1] = f'<a href="{link}" title=""> {info_} </a>'
            new_features = f'{info[-1]} of head= {info[1]}'

        elif feature.startswith('agree'):  # f'agree_{head_pos}_{headrelation.lower()}'
            title = ''
            info_pos, pos_link = get_relation(info[1])
            if len(pos_examples) > 0 and info[1] in pos_examples:
                title = f'For eg. words like {", ".join(pos_examples[info[1]])}'
            info[1] = f'<a href="{pos_link}" title="{title}"> {info[1]} </a>'

            info[-1] = f'<a href="{link}" title=""> {info_} </a>'
            new_features = f'{dependent} agrees with its {info[1]}-head which is a= {info[-1]}'


    if len(info) == 4:
        if feature.startswith('wiki'):
            new_features = f'the head has semantic class= {info[-1]}'

        elif feature.startswith('depfeat'):
            info_pos, pos_link = get_relation(info[1])
            if len(pos_examples) > 0 and info[1] in pos_examples:
                title = f'For eg. words like {", ".join(pos_examples[info[1]])}'
            info[1] = f'<a href="{pos_link}" title="{title}"> {info_pos} </a>'
            new_features = f'{info[1]} is the dependent with {info[2]}= {info[3]}' #f'depfeat_{pos}_{feat}_{value}'

        elif feature.startswith('headfeatrel'):#f'headfeatrel_{rel}_{feat}_{value}'
            info_rel, rel_link = get_relation(info[1])
            info[1] = f'<a href="{rel_link}" title=""> {info_rel} </a>'

            feats_link = 'https://universaldependencies.org/u/feat/index.html'
            info[2] = f'<a href="{feats_link}" title=""> {info} </a>'
            new_features = f'{dependent} has head is a {info[1]} with {info[2]}= {info[3]}'

        elif feature.startswith('headfeat'): #f'headfeat_{head_pos}_{feat}_'
            title = ''
            info_pos, pos_link = get_relation(info[1])
            if len(pos_examples) > 0 and info[1] in pos_examples:
                title = f'For eg. words like {", ".join(pos_examples[info[1]])}'
            info[1] = f'<a href="{pos_link}" title="{title}"> {info_pos} </a>'

            feats_link = 'https://universaldependencies.org/u/feat/index.html'
            info[2] = f'<a href="{feats_link}" title=""> {info[2]} </a>'
            new_features = f'{dependent} is governed by {info[1]} with {info[2]}= {info[3]}'

        elif feature.startswith('headrelrel'):#f'headrelrel_{head_pos}_{relation}_{headrelation.lower()}'
            info_rel, rel_link = get_relation(info[-1])
            info[-1] = f'<a href="{rel_link}" title=""> {info_rel} </a>'

            title = ''
            info_pos, pos_link = get_relation(info[1])
            if len(pos_examples) > 0 and info[1] in pos_examples:
                title = f'For eg. words like {", ".join(pos_examples[info[1]])}'
            info[1] = f'<a href="{pos_link}" title="{title}"> {info_pos} </a>'

            info_rel, rel_link = get_relation(info[2])
            info[2] = f'<a href="{rel_link}" title=""> {info_rel} </a>'
            new_features = f'{dependent} is a {info[2]} of {info[1]} where the head-{info[1]} is a= {info[-1]}'

        elif feature.startswith('head'): #f'head_{pos}_{relation}_{head}'
            info_rel, rel_link = get_relation(info[-1])
            info[-1] = f'<a href="{rel_link}" title=""> {info_rel} </a>'

            info_pos, pos_link = get_relation(info[1])
            if len(pos_examples) > 0 and info[1] in pos_examples:
                title = f'For eg. words like {", ".join(pos_examples[info[1]])}'
            info[1] = f'<a href="{pos_link}" title="{title}"> {info_pos} </a>'

            info_pos, pos_link = get_relation(info[2])
            info[2] = f'<a href="{pos_link}" title=""> {info_pos} </a>'
            new_features = f'{dependent} is a {info[1]} is a {info[2]} of head= {info[-1]}'

        elif feature.startswith('agree'):  # f'agree_{relation}_{head_pos}_{headrelation.lower()}'
            info_rel, rel_link = get_relation(info[-1])
            info[-1] = f'<a href="{rel_link}" title=""> {info_rel} </a>'

            title = ''
            info_pos, pos_link = get_relation(info[1])
            if len(pos_examples) > 0 and info[1] in pos_examples:
                title = f'For eg. words like {", ".join(pos_examples[info[1]])}'
            info[1] = f'<a href="{pos_link}" title="{title}"> {info_pos} </a>'

            info_rel, rel_link = get_relation(info[2])
            info[2] = f'<a href="{rel_link}" title=""> {info_rel} </a>'

            new_features = f'{dependent} is a {info[1]} and agrees with {info[2]}-head which is a= {info[-1]}'

    if len(info) == 5:
        if feature.startswith('headrelrel'):#f'headrelrel_{pos}_{relation}_{head}_{headrelation.lower()}'
            info_rel, rel_link = get_relation(info[-1])
            info[-1] = f'<a href="{rel_link}" title=""> {info_rel} </a>'

            title = ''
            info_pos, pos_link = get_relation(info[1])
            if len(pos_examples) > 0 and info[1] in pos_examples:
                title = f'For eg. words like {", ".join(pos_examples[info[1]])}'
            info[1] = f'<a href="{pos_link}" title="{title}"> {info_pos} </a>'

            title = ''
            info_pos, pos_link = get_relation(info[2])
            if len(pos_examples) > 0 and info[2] in pos_examples:
                title = f'For eg. words like {", ".join(pos_examples[info[2]])}'
            info[1] = f'<a href="{pos_link}" title="{title}"> {info_pos} </a>'

            info_rel, rel_link = get_relation(info[3])
            info[3] = f'<a href="{rel_link}" title=""> {info_rel} </a>'

            new_features =  f'{info[1]} is a {info[2]} of head {info[3]} where that {info[3]} is the= {info[-1]}'

        elif feature.startswith('headfeat'): #f'headfeat_{head_pos}_{relation}_{feat}_{value}'
            info_rel, rel_link = get_relation(info[2])
            info[2] = f'<a href="{rel_link}" title=""> {info_rel} </a>'

            title = ''
            info_pos, pos_link = get_relation(info[1])
            if len(pos_examples) > 0 and info[1] in pos_examples:
                title = f'For eg. words like {", ".join(pos_examples[info[1]])}'
            info[1] = f'<a href="{pos_link}" title="{title}"> {info_pos} </a>'

            feats_link = 'https://universaldependencies.org/u/feat/index.html'
            info[3] = f'<a href="{feats_link}" title=""> {info[3]} </a>'
            new_features = f'{dependent}  is a {info[2]}  of {info[1]} with {info[3]}= {info[4]}'

        elif feature.startswith('depfeat'):  # f'dpefeat_{pos}_{relation}_{feat}_{value}'
            info_rel, rel_link = get_relation(info[2])
            info[2] = f'<a href="{rel_link}" title=""> {info_rel} </a>'

            title = ''
            info_pos, pos_link = get_relation(info[1])
            if len(pos_examples) > 0 and info[1] in pos_examples:
                title = f'For eg. words like {", ".join(pos_examples[info[1]])}'
            info[1] = f'<a href="{pos_link}" title="{title}"> {info_pos} </a>'

            feats_link = 'https://universaldependencies.org/u/feat/index.html'
            info[3] = f'<a href="{feats_link}" title=""> {info[3]} </a>'
            new_features = f'{info[1]} is a {info[2]} with {info[3]}= {info[4]}'


    return new_features

def getHeader(cols, important_features, model, task, relation_map, folder_name, source_lang):
    cols = np.array(cols)
    feats = np.array(important_features)[cols]
    header, subheaders = "", []

    if len(feats) == 1:
        header = ""

        info = transformRulesIntoReadable(feats[0], task, model, relation_map, folder_name, source=source_lang)
        header = info.split("= ")[0]
        subheader = info.split("= ")[-1]

        if "spine" in feats[0]:
            subheader = subheader.split(",")
            subheader = ",".join(subheader[:5]) + "<br>" + ",".join(subheader[5:])

        subheaders.append(subheader)
        return header, subheaders
    for feat in feats:

        info = transformRulesIntoReadable(feat, task, model, relation_map, folder_name, source=source_lang)
        header = info.split("= ")[0]
        subheader = info.split("= ")[-1]
        if "spine" in feat:
            subheader = subheader.split(",")
            subheader = ",".join(subheader[:5]) + "<br>" + ",".join(subheader[5:])

        subheaders.append(subheader)
    return header, subheaders

def iterateTreesFromXGBoost(rules_df_t0, task, model, relation_map, tree_features, folder_name, source_lang):
    topnodes,leafnodes = [],[]
    tree_dictionary = {}
    edge_mapping = {}
    leafedges = defaultdict(list)
    idmap = {}
    for index, row in rules_df_t0.iterrows():
        ID, feature_id, split, yes_id, no_id = row['ID'], row['Feature'], row['Split'], row['Yes'], row['No']
        idmap[ID] = index


    for index, row in rules_df_t0.iterrows():

        ID, feature_id, split, yes_id, no_id = row['ID'], row['Feature'], row['Split'], row['Yes'], row['No']
        split = max(split, 0.0)
        id = idmap[ID]
        if not pd.isna(yes_id):
            yes_id = idmap[yes_id]
        if not pd.isna(no_id):
            no_id = idmap[no_id]

        if id not in tree_dictionary:
            tree_dictionary[id] = {"children": [], "label_distribution": "", "info": "", "id": ID, 'top': -1,
                                        "class": "", "leaf": False, "active": [], "non_active": []}

        if feature_id == 'Leaf':
            tree_dictionary[id]['leaf'] = True
            leafnodes.append(id)  # Leaf node number
        else:
            feature_name = tree_features[int(feature_id.replace('f', ''))]
            feature_label =  transformRulesIntoReadable(feature_name, task, model, relation_map, folder_name, source=source_lang)
            tree_dictionary[id]['edge'] = (feature_name, split)
            edge_mapping[feature_name] = feature_label
            topnodes.append(id)
            tree_dictionary[id]['info'] = (feature_name, split)
            if not pd.isna(yes_id):
                tree_dictionary[id]['children'].append(yes_id)
                #tree_dictionary[id]['yes'] = yes_id
                if yes_id not in tree_dictionary:
                    tree_dictionary[yes_id] = {"children": [], "label_distribution": "", "info": "",
                                        "class": "", "leaf": False, "active": [], "non_active": []}
                tree_dictionary[yes_id]['top'] = id
                tree_dictionary[yes_id]['non_active'].append((feature_name, split)) #if feature < split
                leafedges[yes_id] = feature_name

            if not pd.isna(no_id):
                tree_dictionary[id]['children'].append(no_id)
                #tree_dictionary[id]['no'] = no_id
                if no_id not in tree_dictionary:
                    tree_dictionary[no_id] = {"children": [], "label_distribution": "", "info": "",
                                               "class": "", "leaf": False, "active": [], "non_active": []}
                tree_dictionary[no_id]['top'] = id
                tree_dictionary[no_id]['active'].append((feature_name, split)) #if feature > split
                leafedges[no_id] = 'Not ' + feature_name

    return topnodes, tree_dictionary, leafnodes, leafedges, edge_mapping

def FixLabelTree(dot_data,  tree_dictionary, leafmap, labels, datalabels, threshold=0.01, task='agreement'):
    collatedGraph = dot_data.getvalue().split("\n")
    newGraph, relabeled_leaves = [], {}
    newGraph.append(collatedGraph[0])
    newGraph.append(collatedGraph[1])
    newGraph.append(collatedGraph[2])

    #Get expected distribution
    expected_prob = []
    total = 0
    for label in  labels:
        total += datalabels[label]
    for label in labels:
        expected_prob.append(datalabels[label])


    if len(tree_dictionary) == 1 and len(leafmap) == 1:  #There are no root nodes, possibly one one leaf
        return collatedGraph, leafmap

    updated = {}
    for new_num, leaf in enumerate(leafmap):
        leafinfo = tree_dictionary[leaf]['label_distribution']
        if not  isLabel(expected_prob, leafinfo, threshold, task):
            class_ = tree_dictionary[leaf]['class']
            tree_dictionary[leaf]['class'] = 'NA'

            tree_dictionary[leaf]['info']  = tree_dictionary[leaf]['info'].replace(class_, 'NA')
            updated[leaf]=str(new_num)
    #print('Updated', len(updated))
    for d in collatedGraph[3:]:
        if " ->" in d or '}' in d:
            newGraph.append(d)
        else:
            leafnode=int(d.split(" [")[0].lstrip().rstrip())
            if leafnode in updated:
                new_leaf_info = "{0} [label=\"Leaf- {1}\\n{2}\\lvalue = [{3}]\\lclass = {4}\\l\", fillcolor={5} "
                color = tree_dictionary[leafnode]["info"].split("fillcolor=")[-1]
                label_distribution = [str(l) for l in tree_dictionary[leafnode]['label_distribution']]
                leafinfo = new_leaf_info.format(str(leafnode), str(updated[leafnode]), "", ", ".join(label_distribution),
                                                tree_dictionary[leafnode]['class'], color)
                newGraph.append(leafinfo)
            else:
                newGraph.append(d)


    return newGraph

def FixLabelTreeFromXGBoost(rules_df_t0,  tree_dictionary, leafmap, labels, datalabels, rel, train_df, train_data, threshold=0.01, task='agreement'):
    if len(tree_dictionary) == 1 and len(leafmap) == 1:  #There are no root nodes, possibly one one leaf
        return rules_df_t0, leafmap
    # Get expected distribution
    expected_prob = []
    total = 0
    for label in labels:
        total += datalabels[label]
    for label in labels:
        expected_prob.append(datalabels[label])

    if task == 'agreement':
        class_map = {'0': 'NA', '1': 'req-agree'}
        labels = ['NA', 'req-agree']


    #TODO identify and extract all examples which satisfy the rule and store them [agree, disgaree]
    #If the distrbution of agree-disagree under that leaf is valid then assign the leaf with the class with the label which has larger number of examples
    #first extract active and not active features for each leaf

    updated = {}
    leaf_examples = {}
    leaf_sent_examples = {}
    for new_num, leaf in enumerate(leafmap):

        #sent_examples for each label, examples are not usefule
        examples = {}
        sent_examples = defaultdict(list)
        getExamplesPerLeaf(leaf, tree_dictionary, rel, train_df, train_data, task,
                           examples, sent_examples)
        leaf_examples[leaf] = examples
        leaf_sent_examples[leaf] = sent_examples

        leafinfo, leaflabel, label_distribution = [], {}, []
        for label in labels:
            leafinfo.append(len(sent_examples[label]))
            leaflabel[label] = len(sent_examples[label])
            label_distribution.append(str(len(sent_examples[label])))

        tree_dictionary[leaf]['label_distribution'] = [int(label) for label in label_distribution]
        new_leaf_info = "{0} [label=\"Leaf- {1}\\n{2}\\lvalue = [{3}]\\lclass = {4}\\l\", fillcolor={5} "
        if not  isLabel(expected_prob, leafinfo, threshold, task):
            tree_dictionary[leaf]['class'] = 'NA'
            updated[leaf]=str(new_num)
            class_ = "NA"
        else:
            sorted_leaflabels = sorted(leaflabel.items(), key=lambda kv:kv[1], reverse=True)[0]
            class_ = sorted_leaflabels[0]
            tree_dictionary[leaf]['class'] = class_
        color = colorRetrival(leaflabel, class_)
        tree_dictionary[leaf]['info'] = new_leaf_info.format(str(new_num), str(new_num), "", ", ".join(label_distribution),
                                        tree_dictionary[leaf]['class'], color)


    return leaf_examples, leaf_sent_examples


def FixLabelTreeFromXGBoostWithFeatures(rules_df_t0,  tree_dictionary, leafmap,
                            labels, datalabels, rel,
                            train_df, train_data,
                            tree_features, dataloader, genreparseddata, wikidata, spine_word_vectors, spine_features, spine_dim,
                            threshold=0.01, task='agreement'):
    if len(tree_dictionary) == 1 and len(leafmap) == 1:  #There are no root nodes, possibly one one leaf
        return rules_df_t0, leafmap
    # Get expected distribution
    expected_prob = []
    total = 0
    for label in labels:
        total += datalabels[label]
    for label in labels:
        expected_prob.append(datalabels[label])

    if task == 'agreement':
        labels = ['chance-agree', 'req-agree']
    elif task == 'wordorder':
        labels = ['before', 'after']


    #TODO identify and extract all examples which satisfy the rule and store them [agree, disgaree]
    #If the distrbution of agree-disagree under that leaf is valid then assign the leaf with the class with the label which has larger number of examples
    #first extract active and not active features for each leaf

    updated = {}
    leaf_examples = {}
    leaf_sent_examples = {}
    leafmapcopy = deepcopy(leafmap)
    #train_data = pyconll.load_from_file(train_data_path)
    for new_num, leaf in enumerate(leafmapcopy):

        #sent_examples for each label, examples are not usefule
        examples = {}
        sent_examples = defaultdict(list)
        getExamplesPerLeafWithFeatures(leaf, tree_dictionary, rel, train_df, train_data, task, examples, sent_examples,
                                       tree_features, dataloader, genreparseddata, wikidata, spine_word_vectors,
                                       spine_features, spine_dim)
        # getExamplesPerLeaf(leaf, tree_dictionary, rel, train_df, train_data, task,
        #                    examples, sent_examples)




        leafinfo, leaflabel, label_distribution = [], {}, []
        all_examples_agaree, total = 0, 0
        for label in labels:
            leafinfo.append(len(sent_examples[label]))
            leaflabel[label] = len(sent_examples[label])
            label_distribution.append(str(len(sent_examples[label])))
            total += len(sent_examples[label])
        if total == 0:
            leafmap.remove(leaf)
            continue

        if task == 'agreement':
            all_examples_agaree = leaflabel.get('req-agree')

        leaf_examples[leaf] = examples
        leaf_sent_examples[leaf] = sent_examples
        tree_dictionary[leaf]['label_distribution'] = [int(label) for label in label_distribution]
        new_leaf_info = "{0} [label=\"Leaf- {1}\\n{2}\\lvalue = [{3}]\\lclass = {4}\\l\", fillcolor={5} "
        if not  isLabel(expected_prob, leafinfo, threshold, task):
            if task == 'agreement' and all_examples_agaree > 0:
                tree_dictionary[leaf]['class'] = 'chance-agree'
                updated[leaf] = str(new_num)
                class_ = "chance-agree"
            else: #there are no agreeing examples or class is not significant
                tree_dictionary[leaf]['class'] = 'NA'
                updated[leaf]=str(new_num)
                class_ = "NA"

        else:
            sorted_leaflabels = sorted(leaflabel.items(), key=lambda kv:kv[1], reverse=True)[0]
            class_ = sorted_leaflabels[0]
            tree_dictionary[leaf]['class'] = class_
        color = colorRetrival(leaflabel, class_)
        tree_dictionary[leaf]['info'] = new_leaf_info.format(str(new_num), str(new_num), "", ", ".join(label_distribution),
                                        tree_dictionary[leaf]['class'], color)

    #del train_data
    return leaf_examples, leaf_sent_examples

def FixLabelTreeFromXGBoostWithFeaturesDistributed(rules_df_t0,  tree_dictionary, leafmap,
                            labels, datalabels, rel,
                            train_df_path, train_data_path,
                            tree_features, dataloader,
                            genreparseddata, wikidata, spine_word_vectors, spine_features, spine_dim,
                            threshold=0.01, task='agreement'):

    if len(tree_dictionary) == 1 and len(leafmap) == 1:  #There are no root nodes, possibly one one leaf
        return rules_df_t0, leafmap
    # Get expected distribution
    expected_prob = []
    total = 0
    for label in labels:
        total += datalabels[label]
    for label in labels:
        expected_prob.append(datalabels[label])

    if task == 'agreement':
        labels = ['chance-agree', 'req-agree']
    elif task == 'wordorder':
        labels = ['before', 'after']


    #TODO identify and extract all examples which satisfy the rule and store them [agree, disgaree]
    #If the distrbution of agree-disagree under that leaf is valid then assign the leaf with the class with the label which has larger number of examples
    #first extract active and not active features for each leaf

    filenames = getSmallerFiles(train_data_path)

    updated = {}
    leaf_examples = {}
    leaf_sent_examples = {}
    leafmapcopy = deepcopy(leafmap)

    for new_num, leaf in enumerate(leafmapcopy): #For each leaf, iterate each file

        #sent_examples for each label, examples are not usefule
        examples = {}
        sent_examples = defaultdict(list)
        for file in filenames: #training_data_path is the path to the intermediate path created to hold each file's extracted features
            train_df_path_sub = f'{train_df_path}/{os.path.basename(file)}.train.feats.'

            getExamplesPerLeafWithFeaturesDistributed(leaf, tree_dictionary, rel, train_df_path_sub, file, task, examples, sent_examples,
                                           tree_features, dataloader, genreparseddata, wikidata, spine_word_vectors,
                                           spine_features, spine_dim)

        leafinfo, leaflabel, label_distribution = [], {}, []
        all_examples_agaree, total = 0, 0
        for label in labels:
            leafinfo.append(len(sent_examples[label]))
            leaflabel[label] = len(sent_examples[label])
            label_distribution.append(str(len(sent_examples[label])))
            total += len(sent_examples[label])
        if total == 0:
            leafmap.remove(leaf)
            continue

        if task == 'agreement':
            all_examples_agaree = leaflabel.get('req-agree')

        leaf_examples[leaf] = examples
        leaf_sent_examples[leaf] = sent_examples
        tree_dictionary[leaf]['label_distribution'] = [int(label) for label in label_distribution]
        new_leaf_info = "{0} [label=\"Leaf- {1}\\n{2}\\lvalue = [{3}]\\lclass = {4}\\l\", fillcolor={5} "
        if not  isLabel(expected_prob, leafinfo, threshold, task):
            if task == 'agreement' and all_examples_agaree > 0:
                tree_dictionary[leaf]['class'] = 'chance-agree'
                updated[leaf] = str(new_num)
                class_ = "chance-agree"
            else: #there are no agreeing examples or class is not significant
                tree_dictionary[leaf]['class'] = 'NA'
                updated[leaf]=str(new_num)
                class_ = "NA"

        else:
            sorted_leaflabels = sorted(leaflabel.items(), key=lambda kv:kv[1], reverse=True)[0]
            class_ = sorted_leaflabels[0]
            tree_dictionary[leaf]['class'] = class_
        color = colorRetrival(leaflabel, class_)
        tree_dictionary[leaf]['info'] = new_leaf_info.format(str(new_num), str(new_num), "", ", ".join(label_distribution),
                                        tree_dictionary[leaf]['class'], color)

    return leaf_examples, leaf_sent_examples

def plot_coefficients_label(classifier, feature_names, label_list, top_features=20):
    coef = classifier.coef_
    num_classes = coef.shape[0]
    important_features = {}
    if num_classes > 2:
        for class_ in range(num_classes):
            label = label_list[class_].split("/")[0]
            print(label)
            coefficients = coef[class_,:]
            top_positive_coefficients = np.argsort(coefficients)[-top_features:]
            top_negative_coefficients = np.argsort(coefficients)[:top_features]
            top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
            # # create plot
            # plt.figure(figsize=(15, 5))
            # colors = ['red' if c < 0 else 'blue' for c in coefficients[top_coefficients]]
            # plt.bar(np.arange(2 * top_features), coefficients[top_coefficients], color=colors)

            feature_names = np.array(feature_names)
            required_features = feature_names[top_coefficients][-top_features:]
            print("\n".join([str(r) for r in required_features]))
            # plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
            #plt.show()
            print()

            # plt.savefig(f'./{label}.pdf')
            important_features[class_] = required_features
    else:
        coef = coef.ravel()
        top_positive_coefficients = np.argsort(coef)[-top_features:]
        top_negative_coefficients = np.argsort(coef)[:top_features]
        top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
        # create plot
        # plt.figure(figsize=(15, 5))
        # colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
        # plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
        # feature_names = np.array(feature_names)
        # plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
        #plt.show()
        #plt.savefig(f'./{word}.pdf')

        #Class = 0
        label = label_list[0].split("/")[0]
        print(label)
        required_features = list(feature_names[top_negative_coefficients])
        required_features.reverse()
        print("\n".join([str(r) for r in required_features]) + "\n")
        important_features[0] = required_features

        # Class = 1
        label = label_list[1].split("/")[0]
        print(label)
        required_features = list(feature_names[top_positive_coefficients])
        required_features.reverse()
        print("\n".join([str(r) for r in required_features]))
        important_features[1] = required_features

    return important_features


def collateTree(editedgraph,  tree_dictionary,topnodes, leafnodes, leafedges):
    collatedGraph, relabeled_leaves = [], {}
    if len(tree_dictionary) == 1 and len(leafnodes) == 1:  # There are no root nodes, possibly one one leaf
        for leafnum, leaf in enumerate(leafnodes):
            relabeled_leaves[leafnum] = leaf
        return editedgraph, tree_dictionary, relabeled_leaves


    collatedGraph.append(editedgraph[0])
    collatedGraph.append(editedgraph[1])
    collatedGraph.append(editedgraph[2])
    lableindexmap = {}
    i=0
    removed_leaves = set()
    removednodes = set()


    classm_leaf = {}



    topleafnodes = set()
    for leaf in leafnodes:
        topleafnodes.add(tree_dictionary[leaf]['top'])
    topleafnodes = list(topleafnodes)
    topleafnodes.sort()

    revisedleafnodes = []

    while True:
        #print(i)
        i += 1
        num_changes = 0
        revisedtopleafnodes = defaultdict(set)

        for topnode in topleafnodes: #Traversing only tops of leaves
            if topnode in revisedtopleafnodes:
                continue
            classes = defaultdict(list)
            children = tree_dictionary[topnode]["children"]
            for child_index in children:
                child = tree_dictionary[child_index]
                if child["leaf"]:
                    class_ = child['class']
                    classes[class_].append(child_index)
                num_labels = len(child['label_distribution'])
            for class_, children in classes.items():
                if len(children) > 1:  # Merge the children
                    num_changes += 1
                    labels = []
                    label_distributions = np.zeros((num_labels), )

                    leaf_indices = []
                    for child_index in children:
                        labels.append(tree_dictionary[child_index]["class"])
                        label_distribution = np.array(tree_dictionary[child_index]["label_distribution"])
                        label_distributions += label_distribution
                        leaf_indices.append(child_index)


                    #update the tree_dictionary
                    leaf_label, leaf_index = labels[0], leaf_indices[0]
                    tree_dictionary[leaf_index]["class"] = leaf_label
                    tree_dictionary[leaf_index]["label_distribution"] = label_distributions
                    topedge = tree_dictionary[topnode]['edge']
                    if topedge in tree_dictionary[leaf_index]['active']:
                        tree_dictionary[leaf_index]['active'].remove(topedge)
                    if topedge in tree_dictionary[leaf_index]['non_active']:
                        tree_dictionary[leaf_index]['non_active'].remove(topedge)
                    classm_leaf[leaf_label] = class_

                    for llind in leaf_indices[1:]:
                        tree_dictionary[topnode]['children'].remove(llind)
                        removed_leaves.add(llind)
                        del tree_dictionary[llind]

                    if len(tree_dictionary[topnode]['children']) == 1:
                        child = tree_dictionary[topnode]["children"][0]

                        if topnode == 0:
                            del tree_dictionary[topnode]["children"][0]
                            removednodes.add(child)
                            break
                        else:
                            toptopnode = tree_dictionary[topnode]['top']

                            for active in tree_dictionary[topnode]['active']:
                                tree_dictionary[child]['active'].append(active)
                            for non_active in tree_dictionary[topnode]['non_active']:
                                tree_dictionary[child]['non_active'].append(non_active)

                            tree_dictionary[toptopnode]['children'].append(child)
                            tree_dictionary[child]['top'] = toptopnode
                            del tree_dictionary[topnode]



                            tree_dictionary[toptopnode]['children'].remove(topnode)
                            removednodes.add(topnode)
                            leafedges[child] = leafedges.get(topnode, '')
                            revisedtopleafnodes[toptopnode].add(child)

                    else:
                        revisedtopleafnodes[topnode].add(leaf_index)



                elif len(children) == 1:
                    revisedtopleafnodes[topnode].add(children[0])

        topleafnodes = deepcopy(revisedtopleafnodes)
        if num_changes == 0 or i > 5:
            break

    # for node in topnodes:
    #     #Check if the subtree
    labeltext = "[label=\"{0}\",labelfontsize=10,labelangle={1}];"
    topnodes = set(topnodes) - removednodes

    rules_for_each_leaf = {}

    rule = ""
    root_label = tree_dictionary[0]['class']
    rule_stack = {}

    nonodes = []
    yesnodes= []
    for node in topnodes:
        #if node not in removednodes:
        info = tree_dictionary[node]["info"]
        #new_info = info[0]  + "\\n" + "\\n".join(info[2:])
        collatedGraph.append(info)

        if node == 0:
            rule_stack[node] = ""
        else:
            rule_stack[node] = leafedges[node]
        for num, children in enumerate(tree_dictionary[node]["children"]):
            rule_stack[children] = ""
            textinfo = str(node) + " -> " + str(children)
            angle = 50
            if num > len(tree_dictionary[node]["children"]) / 2:
                angle *= -1


            if "Not" in leafedges[children]:
                edge = "No"
                if tree_dictionary[node]['class'] != tree_dictionary[children]['class']:
                    if node not in rule_stack:
                        print()
                    rule_stack[children] = rule_stack[node]  + leafedges[children] + "\n "
                else:
                    rule_stack[children] = rule_stack[node]
                textinfo += " " + labeltext.format(edge, angle)
                nonodes.append(textinfo)
            else:
                edge = "Yes"
                if node not in rule_stack:
                    print()
                rule_stack[children] = rule_stack[node] + leafedges[children] + "\n "

                textinfo += " " + labeltext.format(edge, angle)
                yesnodes.append(textinfo)


            if children in leafnodes and children not in removed_leaves:  # reached a leaf
                rules_for_each_leaf[children] = rule_stack[children]

    collatedGraph += nonodes + yesnodes

    new_leaf_info = "{0} [label=\"Leaf- {1}\\n{2}\\lvalue = [{3}]\\lclass = {4}\\l\", fillcolor={5} "
    new_leaves, new_leaf_num = {}, 0
    for leaf, leafnode in enumerate(leafnodes):
        if leafnode in removed_leaves:
            continue
        label_distribution = tree_dictionary[leafnode]['label_distribution']
        nsamples = sum(label_distribution)
        label_distribution = [str(l) for l in label_distribution]

        color = tree_dictionary[leafnode]["info"].split("fillcolor=")[-1]
        leafinfo = new_leaf_info.format(leafnode, new_leaf_num, "", ", ".join(label_distribution),tree_dictionary[leafnode]['class'], color)
        collatedGraph.append(leafinfo)

        new_leaves[new_leaf_num] = leafnode
        new_leaf_num += 1

        # if tree_dictionary[leafnode]['class'] != root_label:
        #     print(tree_dictionary[leafnode]['class'] + "\n" + rules_for_each_leaf[leafnode])


    collatedGraph.append(editedgraph[-1])
    #print("\n".join(collatedGraph))
    #exit(-1)
    return collatedGraph, tree_dictionary, new_leaves

def collateTreeFromXGBoost(tree_dictionary,topnodes, leafnodes, leafedges):
    collatedGraph, relabeled_leaves = [], {}
    if len(tree_dictionary) == 1 and len(leafnodes) == 1:  # There are no root nodes, possibly one one leaf
        for leafnum, leaf in enumerate(leafnodes):
            relabeled_leaves[leafnum] = leaf
        return tree_dictionary, relabeled_leaves



    lableindexmap = {}
    i=0
    removed_leaves = set()
    removednodes = set()


    classm_leaf = {}



    topleafnodes = set()
    for leaf in leafnodes:
        top = tree_dictionary[leaf]['top']
        if top == -1:
            continue
        topleafnodes.add(tree_dictionary[leaf]['top'])
    topleafnodes = list(topleafnodes)
    topleafnodes.sort()

    revisedleafnodes = []

    while True:
        #print(i)
        i += 1
        num_changes = 0
        revisedtopleafnodes = defaultdict(set)

        for topnode in topleafnodes: #Traversing only tops of leaves
            if topnode in revisedtopleafnodes:
                continue
            classes = defaultdict(list)
            children = tree_dictionary[topnode]["children"]
            for child_index in children:
                child = tree_dictionary[child_index]
                if child["leaf"]:
                    class_ = child['class']
                    classes[class_].append(child_index)
                num_labels = len(child['label_distribution'])
            for class_, children in classes.items():
                if len(children) > 1:  # Merge the children
                    num_changes += 1
                    labels = []
                    label_distributions = np.zeros((num_labels), )

                    leaf_indices = []
                    for child_index in children:
                        labels.append(tree_dictionary[child_index]["class"])
                        label_distribution = np.array(tree_dictionary[child_index]["label_distribution"])
                        label_distributions += label_distribution
                        leaf_indices.append(child_index)


                    #update the tree_dictionary
                    leaf_label, leaf_index = labels[0], leaf_indices[0]
                    tree_dictionary[leaf_index]["class"] = leaf_label
                    tree_dictionary[leaf_index]["label_distribution"] = label_distributions
                    topedge = tree_dictionary[topnode]['edge']
                    if topedge in tree_dictionary[leaf_index]['active']:
                        tree_dictionary[leaf_index]['active'].remove(topedge)
                    if topedge in tree_dictionary[leaf_index]['non_active']:
                        tree_dictionary[leaf_index]['non_active'].remove(topedge)
                    classm_leaf[leaf_label] = class_

                    for llind in leaf_indices[1:]:
                        tree_dictionary[topnode]['children'].remove(llind)
                        removed_leaves.add(llind)
                        del tree_dictionary[llind]

                    if len(tree_dictionary[topnode]['children']) == 1:
                        child = tree_dictionary[topnode]["children"][0]

                        if topnode == 0 or topnode == -1:
                            del tree_dictionary[topnode]["children"][0]
                            removednodes.add(child)
                            break
                        else:
                            toptopnode = tree_dictionary[topnode]['top']

                            if toptopnode != -1: #Separate tree with only one leaf
                                tree_dictionary[toptopnode]['children'].append(child)
                                tree_dictionary[child]['top'] = toptopnode
                                del tree_dictionary[topnode]
                            else:
                                del tree_dictionary[topnode]["children"][0]
                                removednodes.add(child)
                                del tree_dictionary[topnode]
                                removednodes.add(topnode)
                                break



                            tree_dictionary[toptopnode]['children'].remove(topnode)
                            removednodes.add(topnode)
                            leafedges[child] = leafedges.get(topnode, '')
                            revisedtopleafnodes[toptopnode].add(child)

                    else:
                        revisedtopleafnodes[topnode].add(leaf_index)



                elif len(children) == 1:
                    revisedtopleafnodes[topnode].add(children[0])

        topleafnodes = deepcopy(revisedtopleafnodes)
        if num_changes == 0 or i > 5:
            break

    # for node in topnodes:
    #     #Check if the subtree
    labeltext = "[label=\"{0}\",labelfontsize=10,labelangle={1}];"
    topnodes = set(topnodes) - removednodes

    rules_for_each_leaf = {}
    rule_stack = {}

    nonodes = []
    yesnodes= []
    for node in topnodes:
        #if node not in removednodes:
        info = tree_dictionary[node]["info"]
        #new_info = info[0]  + "\\n" + "\\n".join(info[2:])
        #collatedGraph.append(info)

        if node not in leafedges:
            rule_stack[node] = ""
        else:
            rule_stack[node] = leafedges[node]
        for num, children in enumerate(tree_dictionary[node]["children"]):
            rule_stack[children] = ""
            textinfo = str(node) + " -> " + str(children)
            angle = 50
            if num > len(tree_dictionary[node]["children"]) / 2:
                angle *= -1


            if "Not" in leafedges[children]:
                edge = "No"
                if tree_dictionary[node]['class'] != tree_dictionary[children]['class']:
                    if node not in rule_stack:
                        print()
                    rule_stack[children] = rule_stack[node]  + leafedges[children] + "\n "
                else:
                    rule_stack[children] = rule_stack[node]
                textinfo += " " + labeltext.format(edge, angle)
                nonodes.append(textinfo)
            else:
                edge = "Yes"
                if node not in rule_stack:
                    print()
                rule_stack[children] = rule_stack[node] + leafedges[children] + "\n "

                textinfo += " " + labeltext.format(edge, angle)
                yesnodes.append(textinfo)


            if children in leafnodes and children not in removed_leaves:  # reached a leaf
                rules_for_each_leaf[children] = rule_stack[children]

    #collatedGraph += nonodes + yesnodes

    new_leaf_info = "{0} [label=\"Leaf- {1}\\n{2}\\lvalue = [{3}]\\lclass = {4}\\l\", fillcolor={5} "
    new_leaves, new_leaf_num = {}, 0
    for leaf, leafnode in enumerate(leafnodes):
        if leafnode in removed_leaves:
            continue
        label_distribution = tree_dictionary[leafnode]['label_distribution']
        nsamples = sum(label_distribution)
        label_distribution = [str(l) for l in label_distribution]

        #color = tree_dictionary[leafnode]["info"].split("fillcolor=")[-1]
        #leafinfo = new_leaf_info.format(leafnode, new_leaf_num, "", ", ".join(label_distribution),tree_dictionary[leafnode]['class'], color)
        #collatedGraph.append(leafinfo)

        new_leaves[new_leaf_num] = leafnode
        new_leaf_num += 1

        # if tree_dictionary[leafnode]['class'] != root_label:
        #     print(tree_dictionary[leafnode]['class'] + "\n" + rules_for_each_leaf[leafnode])



    return tree_dictionary, new_leaves

def collateTreeRemoveNA(editedgraph,  tree_dictionary,topnodes, leafnodes, leafedges):
    tree_dictionary_removeNA = deepcopy(tree_dictionary)

    collatedGraph, relabeled_leaves = [], {}
    if len(tree_dictionary_removeNA) == 1 and len(leafnodes) == 1:  # There are no root nodes, possibly one one leaf
        for leafnum, leaf in enumerate(leafnodes):
            relabeled_leaves[leafnum] = leaf
        return editedgraph, tree_dictionary_removeNA, relabeled_leaves


    collatedGraph.append(editedgraph[0])
    collatedGraph.append(editedgraph[1])
    collatedGraph.append(editedgraph[2])

    i=0
    removed_leaves = set()
    removednodes = set()
    classm_leaf = {}
    topleafnodes = set()

    for leaf in leafnodes:
        topleafnodes.add(tree_dictionary_removeNA[leaf]['top'])
    topleafnodes = list(topleafnodes)
    topleafnodes.sort()

    while True:
        i += 1
        num_changes = 0
        revisedtopleafnodes = defaultdict(set)

        for topnode in topleafnodes: #Traversing only tops of leaves
            if topnode in revisedtopleafnodes:
                continue
            classes = defaultdict(list)
            children = tree_dictionary_removeNA[topnode]["children"]
            for child_index in children:
                child = tree_dictionary_removeNA[child_index]
                if child["leaf"]:
                    class_ = child['class']
                    if class_ == 'NA': #Skipping NA class for collating
                        continue
                    classes[class_].append(child_index)
                num_labels = len(child['label_distribution'])
            for class_, children in classes.items():
                if len(children) > 1:  # Merge the children
                    num_changes += 1
                    labels = []
                    label_distributions = np.zeros((num_labels), )

                    leaf_indices = []
                    for child_index in children:
                        labels.append(tree_dictionary_removeNA[child_index]["class"])
                        label_distribution = np.array(tree_dictionary_removeNA[child_index]["label_distribution"])
                        label_distributions += label_distribution
                        leaf_indices.append(child_index)


                    #update the tree_dictionary
                    leaf_label, leaf_index = labels[0], leaf_indices[0]
                    tree_dictionary_removeNA[leaf_index]["class"] = leaf_label
                    tree_dictionary_removeNA[leaf_index]["label_distribution"] = label_distributions
                    topedge = tree_dictionary_removeNA[topnode]['edge']
                    if topedge in tree_dictionary_removeNA[leaf_index]['active']:
                        tree_dictionary_removeNA[leaf_index]['active'].remove(topedge)
                    if topedge in tree_dictionary_removeNA[leaf_index]['non_active']:
                        tree_dictionary_removeNA[leaf_index]['non_active'].remove(topedge)
                    classm_leaf[leaf_label] = class_

                    for llind in leaf_indices[1:]:
                        tree_dictionary_removeNA[topnode]['children'].remove(llind)
                        removed_leaves.add(llind)
                        del tree_dictionary_removeNA[llind]

                    if len(tree_dictionary_removeNA[topnode]['children']) == 1:
                        child = tree_dictionary_removeNA[topnode]["children"][0]

                        if topnode == 0:
                            del tree_dictionary_removeNA[topnode]["children"][0]
                            removednodes.add(child)
                            break
                        else:
                            toptopnode = tree_dictionary_removeNA[topnode]['top']

                            for active in tree_dictionary_removeNA[topnode]['active']:
                                tree_dictionary_removeNA[child]['active'].append(active)
                            for non_active in tree_dictionary_removeNA[topnode]['non_active']:
                                tree_dictionary_removeNA[child]['non_active'].append(non_active)

                            tree_dictionary_removeNA[toptopnode]['children'].append(child)
                            tree_dictionary_removeNA[child]['top'] = toptopnode
                            del tree_dictionary_removeNA[topnode]



                            tree_dictionary_removeNA[toptopnode]['children'].remove(topnode)
                            removednodes.add(topnode)
                            leafedges[child] = leafedges.get(topnode, '')
                            revisedtopleafnodes[toptopnode].add(child)

                    else:
                        revisedtopleafnodes[topnode].add(leaf_index)



                elif len(children) == 1:
                    revisedtopleafnodes[topnode].add(children[0])

        topleafnodes = deepcopy(revisedtopleafnodes)
        if num_changes == 0 or i > 5:
            break

    # for node in topnodes:
    #     #Check if the subtree
    labeltext = "[label=\"{0}\",labelfontsize=10,labelangle={1}];"
    topnodes_ = set(topnodes) - removednodes

    rules_for_each_leaf = {}

    rule = ""
    root_label = tree_dictionary_removeNA[0]['class']
    rule_stack = {}

    nonodes = []
    yesnodes= []
    for node in topnodes_:
        #if node not in removednodes:
        info = tree_dictionary_removeNA[node]["info"]
        #new_info = info[0]  + "\\n" + "\\n".join(info[2:])
        collatedGraph.append(info)

        if node == 0:
            rule_stack[node] = ""
        else:
            rule_stack[node] = leafedges[node]
        for num, children in enumerate(tree_dictionary_removeNA[node]["children"]):
            rule_stack[children] = ""
            textinfo = str(node) + " -> " + str(children)
            angle = 50
            if num > len(tree_dictionary_removeNA[node]["children"]) / 2:
                angle *= -1


            if "Not" in leafedges[children]:
                edge = "No"
                if tree_dictionary_removeNA[node]['class'] != tree_dictionary_removeNA[children]['class']:
                    if node not in rule_stack:
                        print()
                    rule_stack[children] = rule_stack[node]  + leafedges[children] + "\n "
                else:
                    rule_stack[children] = rule_stack[node]
                textinfo += " " + labeltext.format(edge, angle)
                nonodes.append(textinfo)
            else:
                edge = "Yes"
                if node not in rule_stack:
                    print()
                rule_stack[children] = rule_stack[node] + leafedges[children] + "\n "

                textinfo += " " + labeltext.format(edge, angle)
                yesnodes.append(textinfo)


            if children in leafnodes and children not in removed_leaves:  # reached a leaf
                rules_for_each_leaf[children] = rule_stack[children]

    collatedGraph += nonodes + yesnodes

    new_leaf_info = "{0} [label=\"Leaf- {1}\\n{2}\\lvalue = [{3}]\\lclass = {4}\\l\", fillcolor={5} "
    new_leaves, new_leaf_num = {}, 0
    for leaf, leafnode in enumerate(leafnodes):
        if leafnode in removed_leaves:
            continue
        label_distribution = tree_dictionary_removeNA[leafnode]['label_distribution']
        label_distribution = [str(l) for l in label_distribution]

        color = tree_dictionary_removeNA[leafnode]["info"].split("fillcolor=")[-1]
        leafinfo = new_leaf_info.format(leafnode, new_leaf_num, "", ", ".join(label_distribution),tree_dictionary_removeNA[leafnode]['class'], color)
        collatedGraph.append(leafinfo)

        new_leaves[new_leaf_num] = leafnode
        new_leaf_num += 1

    collatedGraph.append(editedgraph[-1])
    return collatedGraph, tree_dictionary_removeNA, new_leaves

def pruneTree(collatedGraph,  tree_dictionary,topnodes, leafnodes):
    newGraph, relabeled_leaves = [], {}
    newGraph.append(collatedGraph[0])
    newGraph.append(collatedGraph[1])
    newGraph.append(collatedGraph[2])
    lableindexmap = {}
    i=0

    removednodes = set()

    if len(tree_dictionary) == 1 and len(leafnodes) == 1:  #There are no root nodes, possibly one one leaf
        return collatedGraph, relabeled_leaves

    classm_leaf = {}



    topleafnodes = set()
    for new_leaf_num, leaf in leafnodes.items():
        top = tree_dictionary[leaf]['top']
        if top == -1:
            continue
        topleafnodes.add(tree_dictionary[leaf]['top'])
    topleafnodes = list(topleafnodes)
    topleafnodes.sort()

    while True:
        #print(i)
        i += 1
        num_changes = 0

        for topnode in topleafnodes: #Traversing only tops of leaves
            if topnode in removednodes:
                continue
            classes = defaultdict(list)
            children = tree_dictionary[topnode]["children"]
            num_children = len(children)
            for child_index in children:
                child = tree_dictionary[child_index]
                if child["leaf"]:
                    class_ = child['class']
                    if class_ == 'NA': #To remove leaf:
                        classes[class_].append(child_index)
            removed_leaves = set()
            for class_, children in classes.items():
                num_changes += 1


                for child_index in children:
                    removed_leaves.add(child_index)
                    del tree_dictionary[child_index]
                    tree_dictionary[topnode]['children'].remove(child_index)
                    removednodes.add(child_index)
                if num_children == len(removed_leaves): #all leaves are 'NA'
                    toptopnode = tree_dictionary[topnode]['top']
                    del tree_dictionary[topnode]
                    tree_dictionary[toptopnode]['children'].remove(topnode)
                    removednodes.add(topnode)


        revisedtopleafnodes = set(topleafnodes) - removednodes
        topleafnodes = deepcopy(revisedtopleafnodes)
        if num_changes == 0 or i > 5:
            break


    leafnum = 0
    new_leaves = {}
    for new_leaf_num, leaf in leafnodes.items():
        if leaf in removednodes:
            continue
        new_leaves[leafnum] = leaf
        leafnum += 1
    return  new_leaves

def collateTreePrune(collatedGraph,  tree_dictionary,topnodes, leafnodes):
    newGraph, relabeled_leaves = [], {}
    newGraph.append(collatedGraph[0])
    newGraph.append(collatedGraph[1])
    newGraph.append(collatedGraph[2])
    lableindexmap = {}
    i=0
    removed_leaves = set()
    removednodes = set()

    if len(tree_dictionary) == 1 and len(leafnodes) == 1:  #There are no root nodes, possibly one one leaf
        return collatedGraph, tree_dictionary, leafnodes

    classm_leaf = {}



    topleafnodes = set()
    for _,leaf in enumerate(leafnodes):
        topleafnodes.add(tree_dictionary[leaf]['top'])
    topleafnodes = list(topleafnodes)
    topleafnodes.sort()

    revisedleafnodes = []

    while True:
        #print(i)
        i += 1
        num_changes = 0
        revisedtopleafnodes = defaultdict(set)

        for topnode in topleafnodes: #Traversing only tops of leaves
            if topnode in revisedtopleafnodes:
                continue
            classes = defaultdict(list)
            children = tree_dictionary[topnode]["children"]
            for child_index in children:
                child = tree_dictionary[child_index]
                if child["leaf"]:
                    class_ = child['class']
                    classes[class_].append(child_index)
                num_labels = len(child['label_distribution'])
            for class_, children in classes.items():
                if len(children) > 1:  # Merge the children
                    num_changes += 1
                    labels = []
                    label_distributions = np.zeros((num_labels), )

                    leaf_indices = []
                    for child_index in children:
                        labels.append(tree_dictionary[child_index]["class"])
                        label_distribution = np.array(tree_dictionary[child_index]["label_distribution"])
                        label_distributions += label_distribution
                        leaf_indices.append(child_index)


                    #update the tree_dictionary
                    leaf_label, leaf_index = labels[0], leaf_indices[0]
                    tree_dictionary[leaf_index]["class"] = leaf_label
                    tree_dictionary[leaf_index]["label_distribution"] = label_distributions
                    topedge = tree_dictionary[topnode]['edge']
                    if topedge in tree_dictionary[leaf_index]['active']:
                        tree_dictionary[leaf_index]['active'].remove(topedge)
                    if topedge in tree_dictionary[leaf_index]['non_active']:
                        tree_dictionary[leaf_index]['non_active'].remove(topedge)
                    classm_leaf[leaf_label] = class_

                    for llind in leaf_indices[1:]:
                        tree_dictionary[topnode]['children'].remove(llind)
                        removed_leaves.add(llind)
                        removednodes.add(llind)
                        del tree_dictionary[llind]

                    if len(tree_dictionary[topnode]['children']) == 1:
                        child = tree_dictionary[topnode]["children"][0]

                        if topnode == 0:
                            del tree_dictionary[topnode]["children"][0]
                            removednodes.add(child)
                            break
                        else:
                            toptopnode = tree_dictionary[topnode]['top']

                            for active in tree_dictionary[topnode]['active']:
                                tree_dictionary[child]['active'].append(active)
                            for non_active in tree_dictionary[topnode]['non_active']:
                                tree_dictionary[child]['non_active'].append(non_active)

                            tree_dictionary[toptopnode]['children'].append(child)
                            tree_dictionary[child]['top'] = toptopnode
                            del tree_dictionary[topnode]



                            tree_dictionary[toptopnode]['children'].remove(topnode)
                            removednodes.add(topnode)
                            revisedtopleafnodes[toptopnode].add(child)

                    else:
                        revisedtopleafnodes[topnode].add(leaf_index)



                elif len(children) == 1:
                    revisedtopleafnodes[topnode].add(children[0])

        topleafnodes = deepcopy(revisedtopleafnodes)
        if num_changes == 0: #or i > 5:
            break

    # for node in topnodes:
    #     #Check if the subtree
    labeltext = "[label=\"{0}\",labelfontsize=10,labelangle={1}];"
    topnodes = set(topnodes) - removednodes


    rule = ""
    root_label = tree_dictionary[0]['class']
    rule_stack = {}

    # nonodes = []
    # yesnodes= []
    # for node in topnodes:
    #     #if node not in removednodes:
    #     info = tree_dictionary[node]["info"]
    #     #new_info = info[0]  + "\\n" + "\\n".join(info[2:])
    #     collatedGraph.append(info)
    #
    #     if node == 0:
    #         rule_stack[node] = ""
    #     else:
    #         rule_stack[node] = leafedges[node]
    #     for num, children in enumerate(tree_dictionary[node]["children"]):
    #         rule_stack[children] = ""
    #         textinfo = str(node) + " -> " + str(children)
    #         angle = 50
    #         if num > len(tree_dictionary[node]["children"]) / 2:
    #             angle *= -1
    #
    #
    #         if "Not" in leafedges[children]:
    #             edge = "No"
    #             if tree_dictionary[node]['class'] != tree_dictionary[children]['class']:
    #                 if node not in rule_stack:
    #                     print()
    #                 rule_stack[children] = rule_stack[node]  + leafedges[children] + "\n "
    #             else:
    #                 rule_stack[children] = rule_stack[node]
    #             textinfo += " " + labeltext.format(edge, angle)
    #             nonodes.append(textinfo)
    #         else:
    #             edge = "Yes"
    #             if node not in rule_stack:
    #                 print()
    #             rule_stack[children] = rule_stack[node] + leafedges[children] + "\n "
    #
    #             textinfo += " " + labeltext.format(edge, angle)
    #             yesnodes.append(textinfo)
    #
    #
    #         if children in leafnodes and children not in removed_leaves:  # reached a leaf
    #             rules_for_each_leaf[children] = rule_stack[children]
    #
    # collatedGraph += nonodes + yesnodes

    new_leaf_info = "{0} [label=\"Leaf- {1}\\n{2}\\lvalue = [{3}]\\lclass = {4}\\l\", fillcolor={5} "
    new_leaves, new_leaf_num = {}, 0
    for leaf, leafnode in enumerate(leafnodes):
        if leafnode in removed_leaves:
            continue
        label_distribution = tree_dictionary[leafnode]['label_distribution']
        nsamples = sum(label_distribution)
        label_distribution = [str(l) for l in label_distribution]

        color = tree_dictionary[leafnode]["info"].split("fillcolor=")[-1]
        leafinfo = new_leaf_info.format(leafnode, new_leaf_num, "", ", ".join(label_distribution),tree_dictionary[leafnode]['class'], color)
        collatedGraph.append(leafinfo)

        new_leaves[new_leaf_num] = leafnode
        new_leaf_num += 1

        # if tree_dictionary[leafnode]['class'] != root_label:
        #     print(tree_dictionary[leafnode]['class'] + "\n" + rules_for_each_leaf[leafnode])


    collatedGraph.append(collatedGraph[-1])
    #print("\n".join(collatedGraph))
    #exit(-1)
    return collatedGraph, tree_dictionary, new_leaves

def isLabel(expected_prob, leafinfo, threshold, task):
    leafinfo = np.array(leafinfo)

    new_leaf_index, new_leaf_info , max_value, max_index=  [],[], 0, -1
    for index, val in enumerate(leafinfo):
        new_leaf_info.append(val)
        new_leaf_index.append(index)
        if val > max_value:
            max_value = val
            max_index = index


    if len(new_leaf_info) < 2:
        return True

    new_leaf_info = np.array(new_leaf_info)
    sumtotal = np.sum(new_leaf_info)
    p_value = max_value / sumtotal #Hard threshold


    expected_prob = np.array(expected_prob)
    total_values = np.sum(expected_prob)

    if task == 'agreement':
        expected_prob_per_feature = 0
        for num in expected_prob:
            p = num / total_values
            expected_prob_per_feature += p * p
        empirical_distr = [1 - expected_prob_per_feature, expected_prob_per_feature]
        expected_agree = empirical_distr[1] * sumtotal
        expected_disagree = empirical_distr[0] * sumtotal
        expected_values = [expected_disagree, expected_agree]
        p_value = new_leaf_info[1] / sumtotal
        k = 1
    else:
        p = 1/ len(expected_prob)
        expected_prob = [p] * len(expected_prob)
        expected_prob = np.array(expected_prob)
        expected_values = expected_prob * sumtotal
        k = len(expected_values)  -1

    min_value = 0
    for value in expected_values:
        if value < 5:
            min_value += 1
    if min_value >=  (k+1) : #Not apply stastical test
        return False


    T,p = chisquare(new_leaf_info, expected_values)
    w = np.sqrt(T * 1.0 /(k * sumtotal))


    if p_value > 0.5 and p < threshold and w > 0.5 : #for majority class label and p-value and effect size and reject the null
        return True
    else:
        return False

def computeAccuracy(tree_dictionary, leafmap, featuresdata, data, prop, labels, task='agreement', threshold=[0.85, 0.90]):
    correct, total, na = 0, 0, 0

    all_data = 0
    leaf_samples = []
    leaf_correct = []
    gold_labels_per_threshold = defaultdict(list)
    leaf_labels = []
    for _,leaf in leafmap.items():
        gold_distribution_leaf = defaultdict(lambda: 0)
        leaf = tree_dictionary[leaf]
        leaf_label = leaf['class']
        top = leaf['top']
        active = leaf['active']
        non_active = leaf['non_active']
        while top > 0:  # Not root
            active += tree_dictionary[top]['active']
            non_active += tree_dictionary[top]['non_active']
            top = tree_dictionary[top]['top']

        active = list(set(active))
        non_active = list(set(non_active))
        features = []
        for index in range(len(featuresdata)):
            datapoint = featuresdata.iloc[index]
            sent_num = int(datapoint['sent_num'])
            token_num = str(datapoint['token_num'])
            sent = data[sent_num]
            for id, token in enumerate(sent):
                if token.id == token_num:
                    feats = token.feats
                    break

            valid = True
            for active_feature in active:
                value = datapoint[active_feature]
                if value != 1:
                    valid = False
                    break
                if not valid:
                    break
                features.append(active_feature)

            if not valid:
                continue

            for non_active_feature in non_active:
                value = datapoint[non_active_feature]
                if value != 0:
                    valid = False
                    break
                if not valid:
                    break
                features.append(non_active_feature)

            if not valid:
                continue

            if prop not in feats:
                continue

            label = list(feats[prop])
            label.sort()
            label = "/".join(label)

            if task == 'agreement':
                headtoken = sent[token.head]
                headfeats = headtoken.feats
                if prop not in headfeats:
                    continue
                headlabel = list(feats[prop])
                headlabel.sort()
                headlabel = '/'.join(headlabel)
                if label == headlabel:
                    gold_distribution_leaf[1] += 1
                else:
                    gold_distribution_leaf[0] += 1
            elif task == 'assignment':
                gold_distribution_leaf[label] += 1
                all_data += 1


        #Gold label as having required agreement > 95%
        leaftotal = 0
        for label in labels:
            leaftotal += gold_distribution_leaf[label]

        if leaftotal == 0:
            continue
        leaf_samples.append(leaftotal)
        total += 1
        leaf_labels.append(leaf_label)

        for t in threshold:
            if task == 'agreement':
                gold_value = gold_distribution_leaf[1] / leaftotal
                if gold_value > t:
                    gold_label = 'req-agree'
                else:
                    gold_label = 'NA'
            elif task == 'assignment':
                majority_label, majorityValue = None, 0
                for label, value in gold_distribution_leaf.items():
                    if value > majorityValue:
                        majorityValue = value
                        majority_label = label
                gold_value = majorityValue / leaftotal
                if gold_value > t:
                    gold_label = majority_label
                else:
                    gold_label = 'NA'
            gold_labels_per_threshold[t].append(gold_label)

    max_score = 0.0
    best_t = 0.0
    leaf_samples = np.array(leaf_samples)
    leaf_distribution = leaf_samples / np.sum(leaf_samples)
    for t, gold_labels in gold_labels_per_threshold.items():
        assert len(leaf_labels) == len(gold_labels)
        leaf_correct = []
        for gold_label, leaf_label in zip(gold_labels, leaf_labels):
            if gold_label == leaf_label:
                correct += 1
                leaf_correct.append(1.0)

            elif leaf_label == 'NA' and gold_label == 'chance-agree':
                correct += 1
                leaf_correct.append(1.0)
            else:
                leaf_correct.append(0.0)
        leaf_distribution = leaf_distribution * np.array(leaf_correct)
        score = np.sum(leaf_distribution)
        if score > max_score:
            max_score = score
            best_t = t


    print(max_score, best_t)
    return best_t

    # acc = correct / total
    # print(acc, correct, total, np.sum(leaf_distribution))

def computeAutomatedMetric(leaves, tree_dictionary, test_df, testdata, feature_pos, tree_features, dataloader, task, genreparseddata, wikidata, spine_word_vectors,
                                       spine_features, spine_dim, foldername, lang, isbaseline=False):
    automated_evaluation_score = {}
    if '-' in feature_pos:
        feature = feature_pos.split("-")[0]
        rel = feature_pos.split("-")[1]
    else:
        feature = feature_pos
        rel = feature_pos

    test_agree_tuple, test_freq_tuple = getTestData(testdata, feature, task)
    examples = {}
    sent_examples = defaultdict(list)

    for leaf_num, leaf in leaves.items(): #test samples fitted on the tree
            getExamplesPerLeafWithFeatures(leaf, tree_dictionary, feature_pos, test_df, testdata, task, examples, sent_examples,
                                       tree_features, dataloader, genreparseddata, wikidata, spine_word_vectors,
                                       spine_features, spine_dim)

    with open(f"{foldername}/{rel}/learnt_rules_{lang}.txt", 'w') as fout:
        fout.write('rel,head-pos,child-pos\ttree-label\tgold-label\n')
        for tuple, total in test_freq_tuple.items():
            agree_examples = test_agree_tuple.get(tuple, 0)

            test_percent = agree_examples * 1.0 / total
            if test_percent >= 0.95:
                gold_label = 'req-agree'
            else:
                gold_label = 'NA'

            #tree-label
            if tuple in examples:
                (rel, head_pos, dep_pos) = tuple
                data_distribution = examples[tuple]
                agree = data_distribution.get('req-agree', 0)
                disagree = data_distribution.get('NA', 0)
                if agree > disagree:
                    tree_label = 'req-agree'
                else:
                    tree_label = 'NA'
                if isbaseline: #all leaves are chance-baseline
                    tree_label = 'NA'
                fout.write(f'{rel},{head_pos},{dep_pos}\t{tree_label}\t{gold_label}\n')
                if tree_label == gold_label:
                    automated_evaluation_score[tuple] = 1
                else:
                    automated_evaluation_score[tuple] = 0

    total = 0
    correct = 0
    for tuple, count in automated_evaluation_score.items():
        correct += count
        total += 1
    metric = correct * 1.0 / total
    return metric, sent_examples

def readWikiData(input):
    wikidata = {}
    with open(input, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            info = line.strip().split("\t")
            babel_id, relation, value = info[0], info[1], int(info[-1])
            if babel_id not in wikidata:
                wikidata[babel_id] = {}
            wikidata[babel_id][relation] = value
    return wikidata

def readWikiParsedInput(input):
    sentence_token_map =defaultdict(list)
    with open(input, 'r') as fin:
        start_token = defaultdict(list)
        for line in fin.readlines():
            if line == "" or line == "\n":
                sentence_token_map[sentence] = deepcopy(start_token)
                start_token = defaultdict(list)
                continue

            elif "bn:" in line:
                info = line.strip().split("\t")

                if "all" in input:
                    token_start, token_end, char_start, char_end, babel_id = info[0], info[1],  info[2], info[3], info[4]
                else:
                    token_start, token_end, char_start, char_end, babel_id = int(info[0]), int(info[1]), int(
                        info[2]), int(info[3]), info[4]
                start_token[token_start].append(babel_id) #Add babelids per each token


            else:
                info = line.strip().split("\t")
                if len(info) == 1:
                    sentence = line.strip()
                elif len(info) == 3:
                    linenum, sentence, tokenid = info[0].lstrip().rstrip(), info[1].lstrip().rstrip(), info[2].lstrip().rstrip()
    return sentence_token_map

def readWSDPath(input):
    word_pos_wsd = {}
    with open(input, 'r') as fin:
        for line in fin.readlines():
            info = line.strip().split("\t")
            word_pos = info[0].split(",")
            features = info[-1].split(";")
            word_pos_wsd[(word_pos[0], word_pos[1])] = features
    return word_pos_wsd

def readWikiGenreInput(input):
    sentence_token_map = {}
    with open(input, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            info = line.strip().split("\t")
            text, tokenids, qids = info[0].lstrip().rstrip(), info[1], info[2]
            tokenids = tokenids.split(",")
            if text not in sentence_token_map:
                sentence_token_map[text] = defaultdict(list)
            sentence_token_map[text][tokenids[0]].append(qids)
    return sentence_token_map

def getTestData(data, feature, task):
    freq_tuple = defaultdict(lambda: 0)
    tuple_agree_information = defaultdict(lambda: 0) #In case of word order agree == right


    for sentence in data:
        id2index = sentence._ids_to_indexes
        for token in sentence:
            token_id = token.id
            if "-" in token_id or  "." in token_id:
                continue
            if token.deprel is None:
                continue
            relation = token.deprel.lower()
            pos = token.upos
            feats = token.feats

            if token.head and token.head != "0":
                head_pos = sentence[token.head].upos
                head_feats = sentence[token.head].feats

                if task == 'agreement':
                    shared, agreed = find_agreement(feats, head_feats)
                    if feature in shared:# and pos == req_pos:
                        tuple = (relation, head_pos, pos)
                        freq_tuple[(relation, head_pos, pos)] += 1
                        if feats[feature] == head_feats[feature]:
                            tuple_agree_information[tuple] += 1
                if task == 'wordorder':
                    feature_info = feature.split("_")
                    if relation == feature_info[0] and head_pos == feature_info[0]:
                        token_position = id2index[token.id]
                        head_position = id2index[token.head]
                        if token_position < head_position:
                            label = 'before'
                        else:
                            label = 'after'
                            tuple_agree_information[feature] += 1
                        freq_tuple[feature] += 1

    return tuple_agree_information, freq_tuple

def getFeaturesForLeaf(leaf, tree_dictionary):
    top = leaf.get('top', -1)

    active = leaf['active']
    non_active = leaf['non_active']

    while top > 0:  # Not root
        active += tree_dictionary[top]['active']
        non_active += tree_dictionary[top]['non_active']
        top = tree_dictionary[top]['top']
    return active, non_active

def filterSents(examples, data, spine_features, isTrain=False, agree_examples=False):
    #each examples (sent_num, token_id, active, non_active)
    sentences_per_unique_token = defaultdict(list)
    filteredExamples = []
    prev_data_path = None
    for (sent_id, token_id, active, non_active) in examples:
        token = data[sent_id][token_id].lemma
        head_id = data[sent_id][token_id].head
        if head_id != '0':
            head = data[sent_id][head_id].lemma
        else:
            head = ""
        sentences_per_unique_token[(token, head)].append((sent_id, token_id, active, non_active))

    for tokenhead, examples in sentences_per_unique_token.items():
        (token, head) = tokenhead
        sent_lengths = {}
        sent_info = {}
        for (sent_id, token_id, active, non_active) in examples:
            sent_lengths[sent_id] = len(data[sent_id]) #Sent length
            sent_info[sent_id] = (token_id, active, non_active)
        #sort sent lenghts
        sorted_sents = sorted(sent_lengths.items(), key=lambda kv:kv[1])[:2]
        #Get at least three examples per each token-head pair
        for (sent_id, length) in sorted_sents:
            token_id, active, non_active = sent_info[sent_id]
            filteredExamples.append((sent_id, token_id, active, non_active))
            active = list(active)
            if agree_examples and len(active) > 0 and len(active[0]) == 2: #len(filteredExamples) < 10:
                for (active_feature, active_value) in active:
                    if isTrain and "spine_" in active_feature:
                        if active_feature not in spine_features:
                            spine_features[active_feature] = defaultdict(lambda: 0)
                        spine_features[active_feature][token.lower()] += active_value

                    if isTrain and "spinehead_" in active_feature:
                        if active_feature not in spine_features:
                            spine_features[active_feature] = defaultdict(lambda: 0)
                        spine_features[active_feature][head.lower()] += active_value
    return filteredExamples

def getExamples(sent_examples, leaf, data, isTrain=False):
    leaf_label = leaf['class']
    found_agree = []
    found_disagree = []
    total_agree, total_disagree = 0,0
    spine_features = {}

    #filter sentences by length and uniqueness of the dependent
    if leaf_label == 'NA':
        all_examples = []
    for task_label, examples in sent_examples.items():
        agree_examples =  leaf_label == task_label
        if leaf_label == 'NA':
            agree_examples = True

        examples = examples[:1000]
        filterExamples = filterSents(examples, data, spine_features, isTrain, agree_examples)
        if leaf_label == 'NA':
            all_examples += filterExamples[:10] #10 examples from each label
            total_disagree += len(filterExamples)
        else:
            if leaf_label == task_label:
                found_agree = filterExamples[:10] #examples[:10]
                total_agree += len(filterExamples)
            else:
                found_disagree = filterExamples[:10] #examples[:10]
                total_disagree += len(filterExamples)

    if leaf_label == 'NA':
        found_disagree = all_examples

    return found_agree, found_disagree, total_agree, total_disagree, spine_features

def getExamplesPerLeaf(leaf, tree_dictionary, prop, train_df, data, task, examples, sent_examples, isTrain=False):
    leaf = tree_dictionary[leaf]
    leaf_label = leaf['class']
    active, non_active = getFeaturesForLeaf(leaf, tree_dictionary)

    #spine_features = {}
    for index in range(len(train_df)):
        #try:
        datapoint = train_df.iloc[index]
        sent_num = int(datapoint['sent_num'])
        #print(sent_num)

        token_num = str(datapoint['token_num'])
        sent = data[sent_num]
        for id, token in  enumerate(sent):
            if token.id == token_num:
                token_id = token.id
                head_token_id = token.head
                head_token = sent[head_token_id]
                feats = token.feats
                relation = token.deprel.lower()
                id_to_indexes = sent._ids_to_indexes
                tuple = (relation, head_token.upos, token.upos)
                break

        if task == 'agreement':
            if prop not in feats or prop not in head_token.feats:
                continue

        valid = True
        for (active_feature, active_value) in active:
            value = float(datapoint[active_feature])
            if value <= active_value:
                valid = False
                break
            if not valid:
                break

            # if isTrain and "spine_" in active_feature:
            #     if active_feature not in spine_features:
            #         spine_features[active_feature] = defaultdict(lambda:0)
            #     spine_features[active_feature][token.form.lower()] += value
            #
            # if isTrain and "spinehead_" in active_feature:
            #     if active_feature not in spine_features:
            #         spine_features[active_feature] = defaultdict(lambda: 0)
            #     spine_features[active_feature][head_token.form.lower()] += value

        if not valid:
            continue

        for (non_active_feature, non_active_value) in non_active:
            value = float(datapoint[non_active_feature])
            if value > non_active_value:
                valid = False
                break
            if not valid:
                break

            # if isTrain and "spine_" in non_active_feature:
            #     if non_active_feature not in spine_features:
            #         spine_features[non_active_feature] = defaultdict(lambda:0)
            #     spine_features[non_active_feature][token.form.lower()] += value
            #
            # if isTrain and "spinehead_" in non_active_feature:
            #     if non_active_feature not in spine_features:
            #         spine_features[non_active_feature] = defaultdict(lambda: 0)
            #     spine_features[non_active_feature][head_token.form.lower()] += value



        if not valid:
            continue

        #Update the spine features, if spine_ then the feature is for dep, if spinehead_ then the feature is for head,
        #collect all tokens which have a high positive value and is frequent in the example

        if task == 'agreement':
            if prop not in feats:
                continue

            label = list(feats[prop])
            label.sort()
            label = "/".join(label)
            task_label = label

            if head_token_id == '0':
                continue
            headlabel = list(head_token.feats[prop])
            headlabel.sort()
            headlabel = "/".join(headlabel)
            if label == headlabel:
                task_label = 'req-agree'
            else:
                task_label = 'chance-agree'
        elif task == 'wordorder':
            if head_token_id == '0':
                continue

            token_position = id_to_indexes[token.id]
            head_position = id_to_indexes[head_token_id]
            if token_position < head_position:
                 label ='before'
            else:
                label = 'after'
            task_label = label
            # if label == leaf_label:
            #     task_label = label
            # else:
            #     task_label = 'NA'
        elif task == 'assignment':
            if token.upos != prop:
                continue
            model = 'Case'
            if model not in feats:
                continue

            label = list(feats[model])
            label.sort()
            label = "/".join(label)
            task_label = label
        elif task.lower() == 'suffix':
            task_label = list(token.misc['affix'])[0]

        if tuple not in examples:
            examples[tuple] = defaultdict(lambda : 0)
        examples[tuple][leaf_label] += 1

        sent_examples[task_label].append((sent_num, token_id, active, non_active))
        #except Exception as e:
        #    print(e, sent_num)


    #return spine_features

def getExamplesPerLeafWithFeatures(leaf, tree_dictionary, prop, train_df, data,
                                   task, examples, sent_examples, tree_features, dataloader,  genreparseddata, wikidata, spine_word_vectors, spine_features, spine_dim):
    leaf = tree_dictionary[leaf]
    leaf_label = leaf['class']
    active, non_active = getFeaturesForLeaf(leaf, tree_dictionary)
    #data = pyconll.load_from_file(data_path)
    #spine_features = {}

    if task == 'agreement':
        if "-" in prop:
            feature_pos = prop.split("-")
            prop = feature_pos[0]
            prop_pos = feature_pos[1]
        else:
            prop_pos = None


    for index in range(len(train_df)):
        #try:
        datapoint = train_df.iloc[index]
        sent_num = int(datapoint['sent_num'])
        #print(sent_num)

        token_num = str(datapoint['token_num'])

        sent = data[sent_num]
        token = sent[token_num]
        token_id = token.id
        head_token_id = token.head
        relation = token.deprel.lower()
        feats = token.feats
        id_to_indexes = sent._ids_to_indexes
        if head_token_id == '0':
            tuple = (relation, token.upos)
            head_token = None
        else:
            head_token = sent[head_token_id]
            tuple = (relation, head_token.upos, token.upos)

        if task == 'agreement':
            if prop not in feats or prop not in head_token.feats:
                continue
            if prop_pos and token.upos != prop_pos:
                continue
        if task.lower() == 'suffix':
            if prop != token.upos:
                continue

        # ensent = endata[sent_num]
        # alignmentsent = alignmentdata[sent_num]
        # alignments_src, alignments_tgt = parseAlignment(alignmentsent.strip().replace("p", "-").split())
        feature_array = dataloader.getFeaturesForToken(sent, token, tree_features, genreparseddata, wikidata, spine_word_vectors, spine_features, spine_dim, prop)
        valid = True
        for (active_feature, active_value) in active:
            value = float(feature_array.get(active_feature, 0))
            if value <= active_value:
                valid = False
                break
            if not valid:
                break

        if not valid:
            continue

        for (non_active_feature, non_active_value) in non_active:
            value = float(feature_array.get(non_active_feature, 0))
            if value > non_active_value:
                valid = False
                break
            if not valid:
                break

        if not valid:
            continue

        #Update the spine features, if spine_ then the feature is for dep, if spinehead_ then the feature is for head,
        #collect all tokens which have a high positive value and is frequent in the example

        if task == 'agreement':
            if prop not in feats:
                continue

            label = list(feats[prop])
            label.sort()
            label = "/".join(label)
            task_label = label

            if head_token_id == '0':
                continue
            headlabel = list(head_token.feats[prop])
            headlabel.sort()
            headlabel = "/".join(headlabel)
            if label == headlabel:
                task_label = 'req-agree'
            else:
                task_label = 'chance-agree'
        elif task == 'wordorder':
            if head_token_id == '0':
                continue

            token_position = id_to_indexes[token.id]
            head_position = id_to_indexes[head_token_id]
            if token_position < head_position:
                 label ='before'
            else:
                label = 'after'
            task_label = label
        elif task == 'assignment':
            if token.upos != prop:
                continue
            model = 'Case'
            if model not in feats:
                continue

            label = list(feats[model])
            label.sort()
            label = "/".join(label)
            task_label = label
        elif task.lower() == 'suffix':
            task_label = list(token.misc['affix'])[0]

        if tuple not in examples:
            examples[tuple] = defaultdict(lambda : 0)
        examples[tuple][leaf_label] += 1

        sent_examples[task_label].append((sent_num, token_id, active, non_active))
        # except Exception as e:
        #     print(e, sent_num)

def getExamplesPerLeafWithFeaturesDistributed(leaf, tree_dictionary, prop, train_df_path, data_path, task, examples, sent_examples, tree_features, dataloader,  genreparseddata, wikidata, spine_word_vectors, spine_features, spine_dim):
    leaf = tree_dictionary[leaf]
    leaf_label = leaf['class']
    active, non_active = getFeaturesForLeaf(leaf, tree_dictionary)
    train_df = pd.read_csv(train_df_path, sep=',')
    data = pyconll.load_from_file(data_path)
    #spine_features = {}
    for index in range(len(train_df)):
        try:
            datapoint = train_df.iloc[index]
            sent_num = int(datapoint['sent_num'])
            #print(sent_num)

            token_num = str(datapoint['token_num'])
            sent = data[sent_num]
            token = sent[token_num]
            token_id = token.id
            head_token_id = token.head
            if head_token_id == '0':
                print()
            head_token = sent[head_token_id]
            feats = token.feats
            relation = token.deprel.lower()
            id_to_indexes = sent._ids_to_indexes
            tuple = (relation, head_token.upos, token.upos)


            if task == 'agreement':
                if prop not in feats or prop not in head_token.feats:
                    continue

            feature_array = dataloader.getFeaturesForToken(sent, token, tree_features, genreparseddata, wikidata, spine_word_vectors, spine_features, spine_dim, prop=prop)
            valid = True
            for (active_feature, active_value) in active:
                value = float(feature_array.get(active_feature, 0))
                if value <= active_value:
                    valid = False
                    break
                if not valid:
                    break

            if not valid:
                continue

            for (non_active_feature, non_active_value) in non_active:
                value = float(feature_array.get(non_active_feature, 0))
                if value > non_active_value:
                    valid = False
                    break
                if not valid:
                    break

            if not valid:
                continue

            #Update the spine features, if spine_ then the feature is for dep, if spinehead_ then the feature is for head,
            #collect all tokens which have a high positive value and is frequent in the example

            if task == 'agreement':
                if prop not in feats:
                    continue

                label = list(feats[prop])
                label.sort()
                label = "/".join(label)
                task_label = label

                if head_token_id == '0':
                    continue
                headlabel = list(head_token.feats[prop])
                headlabel.sort()
                headlabel = "/".join(headlabel)
                if label == headlabel:
                    task_label = 'req-agree'
                else:
                    task_label = 'chance-agree'
            elif task == 'wordorder':
                if head_token_id == '0':
                    continue

                token_position = id_to_indexes[token.id]
                head_position = id_to_indexes[head_token_id]
                if token_position < head_position:
                     label ='before'
                else:
                    label = 'after'
                task_label = label
            elif task == 'assignment':
                if token.upos != prop:
                    continue
                model = 'Case'
                if model not in feats:
                    continue

                label = list(feats[model])
                label.sort()
                label = "/".join(label)
                task_label = label

            if tuple not in examples:
                examples[tuple] = defaultdict(lambda : 0)
            examples[tuple][leaf_label] += 1

            sent_examples[task_label].append((data_path, sent_num, token_id, active, non_active))
        except Exception as e:
            print(e, sent_num)


def example_web_print( ex, outp2, data, task='agreement'):

    try:
        # print('\t\t',data[ex[0]].text)
        sentid = int(ex[0])
        tokid = str(ex[1])
        active  = ex[2]
        # data_path = ex[4]
        # data = pyconll.load_from_file(data_path)
        req_head_head, req_head_dep, req_dep_dep = False, False, False
        for (feature, value) in active:
            if feature.startswith("headrelrel") or feature.startswith('headmatch') or feature.startswith('headhead'):
                req_head_head = True
                break
            if feature.startswith("dephead"):
                req_head_dep = True
            if feature.startswith("depdep"):
                req_dep_dep = True
            info = feature.split("_")
            if len(info) > 2:
                if feature.startswith("agree"):
                    req_head_head = True

        headid = data[sentid][tokid].head
        outp2.write('<pre><code class="language-conllu">\n')
        for token in data[sentid]:
            if token.id == tokid:
                temp = token.conll().split('\t')
                temp[1] = "***" + temp[1] + "***"
                temp2 = '\t'.join(temp)
                outp2.write(temp2 + "\n")
            elif token.id == headid:
                if req_head_head:
                    temp = token.conll().split('\t')
                    if task == 'agreement' or task == 'wordorder':
                        temp[1] = "***" + temp[1] + "***"
                    temp2 = '\t'.join(temp)
                    outp2.write(temp2 + "\n")
                else:
                    temp = token.conll().split('\t')
                    if task == 'agreement' or task == 'wordorder':
                        temp[1] = "***" + temp[1] + "***"

                    temp2 = '\t'.join(temp[:6])
                    outp2.write(f"{temp2}\t0\t_\t_\t_\n")

            elif token.id == data[sentid][headid].head and req_head_head:
                temp = token.conll().split('\t')
                temp2 = '\t'.join(temp[:6])
                outp2.write(f"{temp2}\t0\t_\t_\t_\n")

            elif token.head == headid and req_head_dep: #Getting other dep of the head
                outp2.write(token.conll() + "\n")

            elif tokid == token.head and req_dep_dep:  # Getting other dep of token
                outp2.write(token.conll() + "\n")

            elif '-' not in token.id:
                outp2.write(f"{token.id}\t{token.form}\t_\t_\t_\t_\t0\t_\t_\t_\n")
        outp2.write('\n</code></pre>\n\n')
    # print(f"\t\t\tID: {tokid} token: {data[sentid][tokid].form}\t{data[sentid][tokid].upos}\t{data[sentid][tokid].feats}")
    # print(data[sentid][tokid].conll())
    # headid = data[sentid][tokid].head
    # print(f"\t\t\tID: {headid} token: {data[sentid][headid].form}\t{data[sentid][headid].upos}\t{data[sentid][headid].feats}")
    # print(data[sentid][headid].conll())
    except:
        pass

def example_web_print_with_english(ex, outp2, data, en_data):

    #try:
    # print('\t\t',data[ex[0]].text)
    sentid = int(ex[0])
    tokid = str(ex[1])
    eng_sent = en_data[sentid]
    active = ex[2]
    alignment_sent = ex[3]
    #active  = ex[2] #active features in that examples
    req_head_head, req_head_dep, req_dep_dep = False, False, False

    headid = data[sentid][tokid].head

    for feature in active:
        if feature.startswith("headrelrel") or feature.startswith('headmatch') or feature.startswith('headhead'):
            req_head_head = True
            break
        if feature.startswith("dephead"):
            req_head_dep = True
        if feature.startswith("depdep"):
            req_dep_dep = True
        info = feature.split("_")
        if len(info) > 2:
            if feature.startswith("agree"):
                req_head_head = True

    outp2.write('<pre><code class="language-conllu">\n')
    for token in data[sentid]:
        if token.id == tokid:
            temp = token.conll().split('\t')
            temp[1] = "***" + temp[1] + "*** (" + trn.transform(temp[1]) + " )"
            temp2 = '\t'.join(temp)
            outp2.write(temp2 + "\n")

        elif token.id == headid:
            if req_head_head:
                temp = token.conll().split('\t')
                temp[1] = f'{temp[1]} ({trn.transform(temp[1])})'
                outp2.write("\t".join(temp) + "\n")
            else:
                temp = token.conll().split('\t')
                temp[1] = f'{temp[1]} ({trn.transform(temp[1])})'
                temp2 = '\t'.join(temp[:6])
                outp2.write(f"{temp2}\t0\t_\t_\t_\n")

        elif headid != '0' and token.id == data[sentid][headid].head and req_head_head:
            temp = token.conll().split('\t')
            temp[1] = f'{temp[1]} ({trn.transform(temp[1])})'
            temp2 = '\t'.join(temp[:6])
            outp2.write(f"{temp2}\t0\t_\t_\t_\n")

        elif token.head == headid and req_head_dep: #Getting other dep of the head
            temp = token.conll().split('\t')
            temp[1] = f'{temp[1]} ({trn.transform(temp[1])})'
            outp2.write("\t".join(temp) + "\n")

        elif tokid == token.head and req_dep_dep:  # Getting other dep of token
            temp = token.conll().split('\t')
            temp[1] = f'{temp[1]} ({trn.transform(temp[1])})'
            outp2.write("\t".join(temp) + "\n")

        elif '-' not in token.id:
            temp = token.form
            temp = f'{temp} ({trn.transform(temp)})'
            outp2.write(f"{token.id}\t{temp}\t_\t_\t_\t_\t0\t_\t_\t_\n")
    outp2.write('\n</code></pre>\n\n')


    #Get the English translation
    s2t, t2s =  parseAlignment(alignment_sent.strip().replace("p", "-").split())
    outp2.write('<pre><code class="language-conllu">\n')
    tokid = data[sentid]._ids_to_indexes[tokid]
    for token_num, token in enumerate(eng_sent):
        if token_num in t2s[tokid]: #If the english token is aligned to the main word, mark theb by *** in the english sentence too
            temp = token.conll().split('\t')
            temp[1] = "<b>***" + temp[1] + "***</b>"
            temp2 = '\t'.join(temp)
            outp2.write(temp2 + "\n")

        else:
            outp2.write(f"{token.id}\t{token.form}\t_\t_\t_\t_\t0\t_\t_\t_\n")
    outp2.write('\n</code></pre>\n\n')

    #make a table of all word alignments
    outp2.write(f'<div> <button type="button" class="collapsible" style=\"text-align:center\">Word by word translations </button>\n<div class="content-hover">\n')

    outp2.write(
        f'<table><col><colgroup span="2"></colgroup>'
        f'<th rowspan=\"2\" style=\"text-align:center\">English</th>'
        f'<th rowspan=\"2\" style=\"text-align:center\">Marathi</th></tr><tr>\n')
    for target_id, source_ids in t2s.items():
        target_form = data[sentid][target_id].form
        target_form = target_form + ' (' + trn.transform(target_form) + ')'
        source_forms = []
        for source_id in source_ids:
            source_forms.append(eng_sent[source_id].form)
        source_form = " ".join(source_forms)
        outp2.write(f'<tr><td> {source_form} </td> <td> {target_form} </td></tr>\n')

    outp2.write('</table></div></div>\n<br>\n')

    # except:
    #     pass

def example_web_print_with_transliteration(ex, outp2, data, task='agreement', trn=None):

    #try:
    sentid = int(ex[0])
    tokid = str(ex[1])
    active = ex[2]

    req_head_head, req_head_dep, req_dep_dep = False, False, False
    for (feature, value) in active:
        if feature.startswith("headrelrel") or feature.startswith('headmatch') or feature.startswith('headhead'):
            req_head_head = True
            break
        if feature.startswith("dephead"):
            req_head_dep = True
        if feature.startswith("depdep"):
            req_dep_dep = True
        info = feature.split("_")
        if len(info) > 2:
            if feature.startswith("agree"):
                req_head_head = True

    headid = data[sentid][tokid].head


    outp2.write('<pre><code class="language-conllu">\n')
    for token in data[sentid]:
        if token.id == tokid:
            temp = token.conll().split('\t')
            temp[1] = "***" + temp[1] + "***"
            temp2 = '\t'.join(temp)
            outp2.write(temp2 + "\n")
        elif token.id == headid:
            if req_head_head:
                temp = token.conll().split('\t')
                if task == 'agreement' or task == 'wordorder':
                    temp[1] = "***" + temp[1] + "***"
                temp2 = '\t'.join(temp)
                outp2.write(temp2 + "\n")
            else:
                temp = token.conll().split('\t')
                if task == 'agreement' or task == 'wordorder':
                    temp[1] = "***" + temp[1] + "***"

                temp2 = '\t'.join(temp[:6])
                outp2.write(f"{temp2}\t0\t_\t_\t_\n")

        elif token.id == data[sentid][headid].head and req_head_head:
            temp = token.conll().split('\t')
            temp2 = '\t'.join(temp[:6])
            outp2.write(f"{temp2}\t0\t_\t_\t_\n")

        elif token.head == headid and req_head_dep:  # Getting other dep of the head
            outp2.write(token.conll() + "\n")

        elif tokid == token.head and req_dep_dep:  # Getting other dep of token
            outp2.write(token.conll() + "\n")

        elif '-' not in token.id:
            outp2.write(f"{token.id}\t{token.form}\t_\t_\t_\t_\t0\t_\t_\t_\n")
    outp2.write('\n</code></pre>\n\n')

    outp2.write('<pre><code class="language-conllu">\n')
    for token in data[sentid]:
        transliteration = trn.transform(token.form)
        if token.id == tokid:
            temp = token.conll().split('\t')
            temp[1] = "***" + transliteration + "***"
            temp2 = '\t'.join(temp)
            outp2.write(temp2 + "\n")
        elif token.id == headid:
            if req_head_head:
                temp = token.conll().split('\t')
                if task == 'agreement':
                    temp[1] = "***" + transliteration + "***"
                else:
                    temp[1] = transliteration
                temp2 = '\t'.join(temp)
                outp2.write(temp2 + "\n")
            else:
                temp = token.conll().split('\t')
                if task == 'agreement':
                    temp[1] = "***" + transliteration + "***"
                else:
                    temp[1] = transliteration

                temp2 = '\t'.join(temp[:6])
                outp2.write(f"{temp2}\t0\t_\t_\t_\n")

        elif token.id == data[sentid][headid].head and req_head_head:
            temp = token.conll().split('\t')
            temp[1] = transliteration
            temp2 = '\t'.join(temp[:6])
            outp2.write(f"{temp2}\t0\t_\t_\t_\n")

        elif token.head == headid and req_head_dep:  # Getting other dep of the head
            temp = token.conll().split('\t')
            temp[1] = transliteration
            outp2.write("\t".join(temp) + "\n")

        elif tokid == token.head and req_dep_dep:  # Getting other dep of token
            temp = token.conll().split('\t')
            temp[1] = transliteration
            outp2.write("\t".join(temp) + "\n")

        elif '-' not in token.id:
            outp2.write(f"{token.id}\t{transliteration}\t_\t_\t_\t_\t0\t_\t_\t_\n")
    outp2.write('\n</code></pre>\n\n<br>')

    # except:
    #     pass

def isValidFeature(pos, relation, head_pos,features):
    model_features = []
    for feature_type in features:
        info = feature_type.split("_")
        if info[0] in relation and info[1] == head_pos:
            model_features.append(feature_type)
        if info[0] == pos and info[1] == head_pos:
            model_features.append(feature_type)

    return model_features

def getImportantFeatures(tree_dictionary, leafmap, obs_label, retainNA=False):

    features_involved_labels = []

    for leaf_num in range(len(leafmap)):
        leaf_index = leafmap[leaf_num]
        leaf_label = tree_dictionary[leaf_index]['class']
        if leaf_label != obs_label:
            if not retainNA and leaf_label == 'NA':
                continue
            active = tree_dictionary[leaf_index]['active']  # features which are active for this leaf
            non_active = tree_dictionary[leaf_index]['non_active']  # features which are not active for this leaf
            top = tree_dictionary[leaf_index].get('top', -1)

            if len(active) > 0:
                features_involved_labels += active
            if len(non_active) > 0:
                features_involved_labels += non_active


            while top > 0:  # Not root
                if top not in tree_dictionary:
                    break
                active += tree_dictionary[top]['active']
                non_active += tree_dictionary[top]['non_active']
                top = tree_dictionary[top]['top']
                if len(active) > 0:
                    features_involved_labels += active
                if len(non_active) > 0:
                    features_involved_labels += non_active

    features_involved_labels_only = []
    for (feat, _) in features_involved_labels:
        features_involved_labels_only.append(feat)


    return list(set(features_involved_labels_only)), []
    '''
    while len(queue) > 0:
        topid = queue.pop()
        top = tree_dictionary[topid]
        if 'edge' in top:
            important_features.append(top['edge'][0])

        for children in tree_dictionary[topid]['children']:
            queue.append(children)

    prev_num = -1
    to_combine = [0]
    cols = [to_combine]
    new_important_features = []
    feat_num = 0
    for feat in important_features:
        if feat not in features_involved_labels_only:#Feature is not part of the leaf to be retained
            continue
        if feat in new_important_features:#Avoding repeat columns
            continue
        new_important_features.append(feat)
        if feat_num == 0:
            prev_num = feat_num
            feat_num += 1
            continue
        prev = new_important_features[prev_num]

        prev = prev.split("_")
        feat = feat.split("_")
        if len(prev) == len(feat):
            if "_".join(prev[:-1]) == "_".join(feat[:-1]) and prev[-1] != feat[-1]:
                cols[-1].append(feat_num)
            else:
                to_combine = [feat_num]

                cols.append(to_combine)
        else:
            to_combine = [feat_num]

            cols.append(to_combine)
        prev_num = feat_num
        feat_num += 1
    return new_important_features, cols
    '''

def getColsToCombine(individual_columns):
    prev_num = -1
    to_combine = [0]
    cols = [to_combine]
    new_important_features = []
    feat_num = 0

    individual_columns.sort()

    for feat in individual_columns:
        new_important_features.append(feat)
        if feat_num == 0:
            prev_num = feat_num
            feat_num += 1
            continue
        prev = new_important_features[prev_num]

        prev = prev.split("_")
        feat = feat.split("_")
        if len(prev) == len(feat):
            if "_".join(prev[:-1]) == "_".join(feat[:-1]) and prev[-1] != feat[-1]:
                cols[-1].append(feat_num)
            else:
                to_combine = [feat_num]

                cols.append(to_combine)
        else:
            to_combine = [feat_num]

            cols.append(to_combine)
        prev_num = feat_num
        feat_num += 1
    return cols

def getTreebankPaths(treebank, args):
    train_path, dev_path, test_path, lang = None, None, None, None
    for [path, dir, inputfiles] in os.walk(treebank):
        for file in inputfiles:
            if args.auto:
                if "-auto-ud-train.conllu" in file or "-auto-oscar-train.conllu" in file or "-oscar-auto" in file:
                    train_path = treebank + "/" + file
                    lang = train_path.strip().split('/')[-1].split("-")[0]

                if "-sud-dev.conllu" in file:
                    dev_path = treebank + "/" + file

                if "-sud-test.conllu" in file:
                    test_path = treebank + "/" + file

            else:
                if args.sud:
                    if "-sud-train.conllu" in file and "auto" not in file and "noise" not in file:
                        train_path = treebank + "/" + file
                        lang = train_path.strip().split('/')[-1].split("-")[0]

                    if "-sud-dev.conllu" in file:
                        dev_path = treebank + "/" + file

                    if "-sud-test.conllu" in file:
                        test_path = treebank + "/" + file

                elif args.noise:
                    if "-noise-sud-train.conllu" in file and "auto" not in file:
                        train_path = treebank + "/" + file
                        lang = train_path.strip().split('/')[-1].split("-")[0]

                    if "-sud-dev.conllu" in file:
                        dev_path = treebank + "/" + file

                    if "-sud-test.conllu" in file:
                        test_path = treebank + "/" + file

                else:
                    if "-ud-train.conllu" in file and "auto" not in file:
                        train_path = treebank + "/" + file
                        lang = train_path.strip().split('/')[-1].split("-")[0]

                    if "-ud-dev.conllu" in file:
                        dev_path = treebank + "/" + file

                    if "-ud-test.conllu" in file:
                        test_path = treebank + "/" + file
        break
    print(f'Reading from {train_path}, {dev_path}, {test_path}')
    return train_path, dev_path, test_path, lang

def getTreebankPathsMulti(treebanks, args):
    train_paths, dev_paths, test_paths, langs = [], [], [], []
    treebanks = treebanks.split(",")
    for treebank in treebanks:
        for [path, dir, inputfiles] in os.walk(treebank):
            for file in inputfiles:
                if args.auto:
                    if "-auto-ud-train.conllu" in file or "-auto-oscar-train.conllu" in file or "-oscar-auto" in file:
                        train_path = treebank + "/" + file
                        lang = train_path.strip().split('/')[-1].split("-")[0]

                    if "-sud-dev.conllu" in file:
                        dev_path = treebank + "/" + file

                    if "-sud-test.conllu" in file:
                        test_path = treebank + "/" + file

                else:
                    if "-sud-train.conllu" in file and "auto" not in file:
                        train_path = treebank + "/" + file
                        lang = train_path.strip().split('/')[-1].split("-")[0]
                        train_paths.append(train_path)
                        langs.append(lang)

                    if "-sud-dev.conllu" in file:
                        dev_path = treebank + "/" + file
                        dev_paths.append(dev_path)

                    if "-sud-test.conllu" in file:
                        test_path = treebank + "/" + file
                        test_paths.append(test_path)

    #print(f'Reading from {train_path}, {dev_path}, {test_path}')
    return train_paths, dev_paths, test_paths, langs

def colorRetrival(labeldistribution, class_):
    agreement_color_schemes = {0.1: '#eff3ff', 0.5: '#bdd7e7', 0.9: '#2171b5'}
    chanceagreement_color_schemes = {0.1: '#fee8c8', 0.5: '#fdbb84', 0.9: '#e34a33'}

    total = 0
    max_value, max_label = 0, ''
    for label, value in labeldistribution.items():
        if value > max_value:
            max_label = label
            max_value = value
        total += value

    t = max_value / total
    if class_ != 'NA':#t >= threshold:
        if t >= 0.9:
            color = agreement_color_schemes[0.9]
        elif t >= 0.5:
            color = agreement_color_schemes[0.5]
        else:
            color = agreement_color_schemes[0.1]
    else:
        if (1 - t) >= 0.9:
            color = chanceagreement_color_schemes[0.9]
        elif (1 - t) >= 0.5:
            color = chanceagreement_color_schemes[0.5]
        else:
            color = chanceagreement_color_schemes[0.1]
    return color

def getWikiFeatures(args, lang, test=False):
    genre_train_data, genre_dev_data, genre_test_data, wikiData = None, None, None, None
    if args.use_wikid:
        # Get the wikiparsed data
        wikiData = readWikiData(args.wikidata)
        if not test:
            genre_train_file = args.wiki_path + f'{lang}_train_genre_output_formatted.txt'
            genre_dev_file = args.wiki_path + f'{lang}_dev_genre_output_formatted.txt'
            genre_test_file = args.wiki_path + f'{lang}_test_genre_output_formatted.txt'

            if os.path.exists(genre_train_file) and os.path.exists(genre_test_file):
                genre_train_data = readWikiGenreInput(genre_train_file)
                if os.path.exists(genre_dev_file):
                    genre_dev_data = readWikiGenreInput(genre_dev_file)
                genre_test_data = readWikiGenreInput(genre_test_file)
        else:
            genre_test_file = args.wiki_path + f'{lang}_test_genre_output_formatted.txt'
            if os.path.exists(genre_test_file):
                genre_test_data = readWikiGenreInput(genre_test_file)

    return genre_train_data, genre_dev_data, genre_test_data, wikiData

def getBabelNetFeatures(args, lang, treebank):
    # Working with BabelNet annotated
    wiki_train_sentence_token_map, wiki_dev_sentence_token_map, wiki_test_sentence_token_map = None, None, None
    word_pos_wsd, lemma_pos_wsd = None, None
    use_prefix_all = False
    if False: #args.use_animate:
        wiki_train_file = args.wiki_path + f'{lang}_train.txt'
        wiki_test_file = args.wiki_path + f'{lang}_test.txt'
        wiki_dev_file = args.wiki_path + f'{lang}_dev.txt'
        wiki_wsd = treebank + f'/{lang}-wsd.txt'
        wiki_wsd_lem = treebank + f'/{lang}-wsd-l.txt'
        wiki_train_sentence_token_map = readWikiParsedInput(wiki_train_file)
        wiki_test_sentence_token_map = readWikiParsedInput(wiki_test_file)

        if "_all" in wiki_train_file:
            use_prefix_all = True
        if os.path.exists(wiki_dev_file):
            wiki_dev_sentence_token_map = readWikiParsedInput(wiki_dev_file)

        if args.lp and os.path.exists(wiki_wsd):
            word_pos_wsd =readWSDPath(wiki_wsd)
            if args.lp and os.path.exists(wiki_wsd_lem):
                lemma_pos_wsd =readWSDPath(wiki_wsd_lem)
    return wiki_train_sentence_token_map, wiki_dev_sentence_token_map, wiki_test_sentence_token_map, word_pos_wsd, lemma_pos_wsd , use_prefix_all

def reloadData(train_file, dev_file, test_file):
    train_df = pd.read_csv(train_file, sep=',')
    dev_df = None
    if os.path.exists(dev_file):
        dev_df = pd.read_csv(dev_file, sep=',')
    test_df = pd.read_csv(test_file, sep=',')
    columns = train_df.columns.to_list()
    return train_df, dev_df, test_df, columns

def reloadDataLibsvm(train_file, dev_file, test_file):
    dtrain = xgb.DMatrix(train_file)
    dev_df = None
    if os.path.exists(dev_file):
        ddev = xgb.DMatrix(dev_file)
    dtest = xgb.DMatrix(test_file)
    columns = dtrain.feature_names
    return dtrain, ddev, dtest, columns

def getModelTrainingData(train_df, dev_df, test_df):
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(train_df[["label"]])
    label2id = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    id2label = {v: k for k, v in label2id.items()}
    label_list = []
    for i in range(len(id2label)):
        label_list.append(id2label[i])

    train_features, train_label = train_df.drop(columns=['label', 'sent_num', 'token_num'],
                                                axis=1), label_encoder.transform(train_df[["label"]])

    dev_features, dev_label = None, None
    if dev_df is not None and not dev_df.empty:
        dev_features, dev_label = dev_df.drop(columns=['label', 'sent_num', 'token_num'],
                                              axis=1), label_encoder.transform(
            dev_df[["label"]])

    test_features, test_label = test_df.drop(columns=['label', 'sent_num', 'token_num'],
                                             axis=1), label_encoder.transform(
        test_df[["label"]])

    most_fre_label = train_df['label'].value_counts().max()
    trainlabels = train_df['label'].value_counts().index.to_list()
    trainvalues = train_df['label'].value_counts().to_list()
    minority, majority = 1000000, 0
    for label, value in zip(trainlabels, trainvalues):
        if value == most_fre_label:
            baseline_label = label
            majority = value
        if value < minority:
            minority = value

    return train_features, train_label, dev_features, dev_label, test_features, test_label, baseline_label, id2label, label_list, label_encoder, minority, majority

def getModelTrainingDataLibsvm(train_file, dev_file, test_file, label2id, datalabels):
    if os.path.exists(dev_file):
        all_data = load_svmlight_files([train_file, dev_file, test_file], zero_based=True)
        train_features, train_labels = all_data[0], all_data[1]
        dev_features, dev_labels = all_data[2], all_data[3]
        test_features, test_labels = all_data[4], all_data[5]
    else:
        all_data = load_svmlight_files([train_file, test_file], zero_based=True)
        train_features, train_labels = all_data[0], all_data[1]
        dev_features, dev_labels = None, None
        test_features, test_labels = all_data[2], all_data[3]

    counter = Counter(train_labels)
    minority, majority = 1000000, 0
    baseline_label = -1
    id2label = {v: k for k, v in label2id.items()}

    values = []
    for class_ in range(len(counter)):
        value = counter[class_]
        datalabels[id2label[class_]] = value

        values.append(f'{id2label[class_]}: {value}')

        if value < minority:
            minority = value
        if value > majority:
            majority = value
            baseline_label = id2label[class_]

    print(",".join(values))

    return train_features, train_labels , dev_features, dev_labels, test_features, test_labels, baseline_label, id2label, minority, majority

def removeFeatures(train_df, t=0):
    valid = True
    train_features, train_label_info = train_df.drop(columns=['label', 'sent_num', 'token_num'],
                                                     axis=1), train_df[['label', 'sent_num', 'token_num']]

    sel = VarianceThreshold(threshold=(t * (1-t)))
    columns = np.array(train_features.columns.to_list())
    orig_features = np.array(train_features.columns.to_list())

    train_features = sel.fit_transform(train_features)
    columns = columns[sel.get_support()]

    print('Before', len(orig_features), 'After', len(columns))
    if len(columns) == 0:
        valid = False
    updated_columns = np.concatenate((columns, ['label', 'sent_num', 'token_num']))
    #updated_columns = np.concatenate((orig_features, ['label', 'sent_num', 'token_num']))
    return updated_columns, valid

def loadSpine(spine_outputs, lang, is_continuous=False):
    word_vectors = {}
    feats = []

    spine_emb = f'{spine_outputs}/{lang}_spine.txt'
    spine_feats = f'{spine_outputs}/{lang}_spine_feats_ud_freqtop.txt'
    spine_feats = f'{spine_outputs}/{lang}_spine_feats_knen.txt'
    with open(spine_emb, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            info = line.strip().split()
            word = info[0].lower()
            embs = [abs(float(e)) for e in info[1:]]
            if is_continuous:
                word_vectors[word] = np.array(embs)
            else:
                new_embs = []
                for e in embs:
                    if e < 0.9:
                        new_embs.append(0)
                    else:
                        new_embs.append(1)
                word_vectors[word] = np.array(new_embs)
    with open(spine_feats, 'r') as fin:
        for line in fin.readlines():
            info = line.strip().split("\t")
            if len(info) > 2:
                if info[1] != 'NA':
                    feats.append(info[1])
                else:
                    feats.append((info[-1]))
            else:
                feats.append(info[-1])
    assert len(embs) == len(feats)
    return word_vectors, np.array(feats), len(embs)

def transformSpine(tree_rules, spine_features, spine_word_vectors, train_data, dot_data):
    dot_data = dot_data.getvalue().split("\n")
    spine_features = list(spine_features)
    feature_value = {}
    for feature in tree_rules:
        if "<=" in feature:
            feature_info = feature.split("--- ")[-1]
            feature, value = feature_info.split("<=")[0].lstrip().rstrip(), float(feature_info.split("<=")[1].lstrip().rstrip())
            feature_value[feature] = value

    dim_indices, spine_features_of_interest, values = [], [], []
    for feature, value in feature_value.items():
        if feature.startswith('spine'):
            feature = feature.replace("spinehead_", "").replace("spine_", "")
            dim_indices.append(spine_features.index(feature))
            spine_features_of_interest.append(feature)
            values.append(value)

    dim_indices, values = np.array(dim_indices), np.array(values)
    new_features = {}
    new_feature_names = {}
    if len(dim_indices) > 0:
        for sent in train_data:
            for token in sent:
                if token:
                    form = token.form.lower()
                    if token.upos not in ['ADJ', 'NOUN', 'PROPN', 'INTJ', 'ADV', 'VERB']:
                        continue
                    if form in spine_word_vectors:
                        vector = spine_word_vectors[form]
                        dim_values = vector[dim_indices] #Get the spine values for the spine index of interest
                        for dim_value, threshold_value, old_feature in zip(dim_values, values, spine_features_of_interest):
                            if old_feature not in new_features:
                                new_features[old_feature] = defaultdict(lambda : 0)
                            if dim_value >= threshold_value: #
                                new_features[old_feature][form] = max(dim_value, new_features[old_feature].get(form, 0.0))

        for old_feature, new_feature_info in new_features.items():
            sorted_features = sorted(new_feature_info.items(), key=lambda kv: kv[1], reverse=True)[:5]
            header = []
            for (f, _) in sorted_features:
                header.append(f)
            if len(header) > 0:
                new_feature_names[old_feature] = ",".join(header)
            else:
                new_feature_names[old_feature] = old_feature


    if len(new_feature_names) > 0:
        new_dot_data = []
        for d in dot_data:
            if 'spine' in d:
                for old, new in new_feature_names.items():
                    if old in d:
                        d = d.replace(old, new)
                        new_dot_data.append(d)
                        break
            else:
                new_dot_data.append(d)
            #print(d)
    return new_feature_names, new_dot_data

def getActiveNotActive(active, non_active, columns_to_retain, task, rel, relation_map, folder_name, source_lang):
    active_text, nonactive_text = "Active:", "Not Active:"
    if len(active) > 0:
        covered_act = set()
        for a in active:
            if a in covered_act or a not in columns_to_retain:
                continue
            active_human = a #transformRulesIntoReadable(a, task, rel, relation_map, folder_name, source=source_lang)
            covered_act.add(a)
            active_text += active_human + ";"


    if len(non_active) > 0:
        covered_na = set()
        for n in non_active:
            if n in covered_na or n not in columns_to_retain:
                continue
            nonactive_human = n #transformRulesIntoReadable(n, task, rel, relation_map, folder_name, source=source_lang)
            covered_na.add(n)
            nonactive_text += nonactive_human + ";"

    return active_text, nonactive_text

def createRuleTable(outp2, active, non_active, task, rel, relation_map, columns_to_retain, folder_name, source_lang):
    # Print features active/inactive in the rule
    active_text, nonactive_text = "Active:", "Not Active:"
    outp2.write("<div id=\"rules\">")
    outp2.write(
        f'<table><col><colgroup span=\"2\"></colgroup>'
        f'<tr><th colspan=\"2\"	scope=\"colgroup\" style=\"text-align:center\">Features that make up this rule</th></tr>'
        f'<th rowspan=\"1\" style=\"text-align:center\">Active Features</th>'
        f'<th rowspan=\"1\" style=\"text-align:center\">Inactive Features</th>'
        f'</tr><tr>\n')

    if len(active) > 0:
        outp2.write(f'<td style=\"text-align:center\">')
        covered_act = set()
        for a in active:
            if a in covered_act or a not in columns_to_retain:
                continue
            active_human = transformRulesIntoReadable(a, task, rel, relation_map, folder_name, source=source_lang)
            outp2.write(f'{active_human} <br>\n')
            covered_act.add(a)
            active_text += a + ";"
        outp2.write(f'</td>')
    else:
        outp2.write(f'<td style=\"text-align:center\"> - </td>\n')

    if len(non_active) > 0:
        outp2.write(f'<td style=\"text-align:center\">')
        covered_na = set()
        for n in non_active:
            if n in covered_na or n not in columns_to_retain:
                continue
            nonactive_human = transformRulesIntoReadable(n, task, rel, relation_map, folder_name, source=source_lang)
            covered_na.add(n)
            nonactive_text += n + ";"
            outp2.write(f'{nonactive_human} <br>\n')
        outp2.write(f'</td>')
    else:
        outp2.write(f'<td style=\"text-align:center\"> - </td>\n')
    outp2.write(f'</tr></table></div>\n')


    outp2.write(f'<input type="hidden" id="rules_text" name="rules_text" value="{active_text + " " + nonactive_text}">')

def createRuleTableSuffix(outp2, active, active_features, task, rel, relation_map, folder_name, label, pos_examples, source='eng'):
    # Print features active/inactive in the rule
    active_text, nonactive_text = "Active:", "Not Active:"
    outp2.write("<div id=\"rules\">")
    outp2.write(
        f'<table><col><colgroup span=\"2\"></colgroup>'
        f'<tr><th colspan=\"2\"	scope=\"colgroup\" style=\"text-align:center\">Features that make up this rule</th></tr>'
        #f'<th rowspan=\"1\" style=\"text-align:center\">Active Features</th>'
        #f'<th rowspan=\"1\" style=\"text-align:center\">Inactive Features</th>'
        f'</tr><tr>\n')

    if len(active) > 0:
        outp2.write(f'<td style=\"text-align:center\">')
        covered_act = set()
        for a in active:
            if a in covered_act and a not in active_features:
                continue
            active_human = transformRulesIntoReadable(a, task, rel, relation_map, folder_name, pos_examples, source)
            outp2.write(f'{active_human} <br>\n')
            covered_act.add(a)
            active_text += active_human + ";"
        outp2.write(f'</td>')
    else:
        outp2.write(f'<td style=\"text-align:center\"> - </td>\n')

    outp2.write(f'</tr></table></div>\n')
    outp2.write(f'<input type="hidden" id="rules_text_{label}" name="rules_text">')

def printExamples(outp2, data, agree, disagree, dep, leaf_label, task, source_lang):
    outp2.write(f'<div id=\"examples\">\n')

    if len(agree) > 0:
        if leaf_label == 'NA':
            outp2.write(f'<h4> Examples: The words of interest are denoted by *** </h4>')
        elif leaf_label == 'chance-agree':
            outp2.write(f'<h4> Examples that do not agree: The words of interest are marked by ***, hover over those tokens to see more information. </h4>')
        else:
            outp2.write(f'<h4> Examples that agree with label: <b>{leaf_label}</b>: The tokens of interest are denoted by ***, hover over those tokens to see more information. </h4>')

        for eg in agree:
            if source_lang and source_lang != 'eng':
                trn = Transliterator(source=source_lang, target='eng', build_lookup=True)
                example_web_print_with_transliteration(eg, outp2, data, task, trn)
            else:
                example_web_print(eg, outp2, data, task)
    else:
        if leaf_label != 'NA' and leaf_label != 'chance-agree':
            outp2.write(f'<h5> No examples found </h5>\n')


    if len(disagree) > 0:
        if leaf_label == 'NA':
            outp2.write(f'<h4> Examples: The words of interest are denoted by *** </h4>')

        elif leaf_label == 'chance-agree':
            outp2.write(f'<h4> Examples that agree: The words of interest are denoted by *** </h4>')

        else:
            outp2.write(f'<h4> Examples that disagree with the label: <b>{leaf_label}</b> </h4>\n')

        for eg in disagree:
            if source_lang and source_lang != 'eng':
                trn = Transliterator(source=source_lang, target='eng', build_lookup=True)
                example_web_print_with_transliteration(eg, outp2, data, task, trn)
            else:
                example_web_print(eg, outp2, data, task)
    else:
        outp2.write(f'<h5> No examples found </h5>\n')

    outp2.write(f'</div>\n')

def populateLeaf(data, leaf_examples_file, leaf_label, agree, disagree,
                 active, non_active,
                 default_leaf, rel, task, language_full_name, relation_map, columns_to_retain, ORIG_HEADER, HEADER, FOOTER, folder_name, source_lang):
    # Populate the leaf with examples
    with open(leaf_examples_file, 'w') as outp2:
        if task == 'agreement':
            HEADER = ORIG_HEADER.replace("main.css", "../../../../main.css")
        else:
            HEADER = ORIG_HEADER.replace("main.css", "../../../main.css")

        outp2.write(HEADER + '\n')
        if task == 'agreement':
            outp2.write(
            f'<ul class="nav"><li class="nav"><a class="active" href=\"../../../../index.html\">Home</a>'
            f'</li><li class="nav"><a href=\"../../../../introduction.html\">Usage</a></li>'
            f'<li class="nav"><a href="../../../../about.html\">About Us</a></li></ul>')
        else:
            outp2.write(
            f'<ul class="nav"><li class="nav"><a class="active" href=\"../../../index.html\">Home</a>'
            f'</li><li class="nav"><a href=\"../../../introduction.html\">Usage</a></li>'
            f'<li class="nav"><a href="../../../about.html\">About Us</a></li></ul>')
        outp2.write(
            f"<br><li><a href=\"{rel}.html\">Back to {language_full_name} page</a></li>\n")


        if task == 'wordorder':
            modelinfo = rel.split("-")

            if leaf_label == 'NA':
                outp2.write(
                f'<h3> Cannot decide the relative order of <b>{modelinfo[0]}</b> with its  head <b>{modelinfo[1]}</b> </h3>')
            else:
                outp2.write(
                f'<h3> <b>{modelinfo[0]}</b> is  <b>{leaf_label}</b> its  head <b>{modelinfo[1]}</b> </h3>')
            dep =  modelinfo[0]

        elif task == 'assignment':
            modelinfo = rel
            if leaf_label == 'NA':
                outp2.write(
                    f'<h3> Cannot decide the case  of <b>{modelinfo}</b> </h3>')
            else:
                outp2.write(
                    f'<h3> <b>{modelinfo}</b> has Case  <b>{leaf_label}</b> </h3>')
            dep = rel

        elif task == 'agreement':
            if "-" in rel:
                feature = rel.split("-")[0]
                pos = rel.split("-")[1]
            else:
                feature = rel
                pos = "dependent"
            if leaf_label == 'chance-agree':
                outp2.write(
                    f'<h3> There is <b> NO required-agreement </b> between the  head and its {pos} for <b>{feature}</b> </h3>')

            elif leaf_label == 'req-agree':
                outp2.write(
                    f'<h3> The {rel} values <b> should match </b> between the  head and its {pos} </h3>')

            dep = "dependent"
        elif task.lower() == 'suffix':
            dep = "current word"

        # output the rule in each leaf
        if default_leaf:
            active_text, nonactive_text = getActiveNotActive(active, non_active, columns_to_retain, task, rel, relation_map, folder_name, source_lang)
            outp2.write(
                f'<input type="hidden" id="rules_text" name="rules_text" value="{active_text + " " + nonactive_text}">')
            #outp2.write("<h2> Examples for the default label </h3> ")
        else:
            createRuleTable(outp2, active, non_active, task, rel, relation_map, columns_to_retain, folder_name, source_lang)

        # print examples in each leaf
        printExamples(outp2, data, agree, disagree,dep, leaf_label, task, source_lang)

        outp2.write("</ul><br><br><br>\n" + FOOTER + "\n")

def getSmallerFiles(one_file):
    dir = os.path.dirname(one_file)
    filenames = []
    for [path, _, files] in os.walk(dir):
        for file in files:
            if file.endswith("conllu"):
                filenames.append(dir + "/" + file)
    print(f'Loading smaller files from {dir}: {len(filenames)}')
    return filenames

def parseAlignment(info):
    src_tgt_alignments = defaultdict(list)
    tgt_src_alignments = defaultdict(list)
    for align in info:
        s = int(align.split('-')[0])
        t = int(align.split('-')[1])
        src_tgt_alignments[s].append(t)
        tgt_src_alignments[t].append(s)
    return src_tgt_alignments, tgt_src_alignments

