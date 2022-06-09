import pyconll
import utils
import numpy as np
np.random.seed(1)
from collections import defaultdict
import codecs
import os
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from copy import deepcopy
import random
import json
import pandas as pd
from indictrans import Transliterator

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

def checkTransitivity(verb_token, tokens):
    is_obj = False
    for token in tokens:
        if token.head != '0' and token.head == verb_token.id: #If a token's head is the given token
            if token.deprel is not None and token.deprel == 'obj':
                is_obj = True
                break
        if is_obj:
            break
    if is_obj:
        return True
    return False



class DataLoader(object):
    def __init__(self, args, relation_map):
        self.args = args
        self.feature_dictionary = defaultdict(lambda : 0)
        self.feature_map = {}
        self.labels = defaultdict(lambda : 0)
        self.triple_freq = defaultdict(lambda: 0)
        self.feature_freq = {}
        self.required_features = []#['Tense', 'Animacy', 'Aspect', 'Mood', 'Definite', 'Voice', 'Poss','PronType', 'VerbForm']
        self.required_relations = ['obj', 'mod', 'det', 'mod', 'subj', 'vocative', 'aux', 'compound', 'conj', 'flat', 'appos']  # ,
        #self.required_relations = []
        self.remove_features = ['Gender', 'Person', 'Number']
        self.model_data_case = {}
        self.model_data = defaultdict(lambda : 0)
        self.model_dictionary = {} #training one model per each property and each POS
        self.model_feature_label = {}
        self.examples_per_pos = {}
        random.seed(args.seed)

    def unison_shuffled_copies(self,a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def isValidLemma(self, lemma, upos):
        if upos in ['PUNCT', 'NUM', 'PROPN' 'X', 'SYM']:
            return None
        if lemma:
            lemma = lemma.lower()
            lemma = lemma.replace("\"", "").replace("\'", "")
            if lemma == "" or lemma == " ":
                return None
            else:
                return lemma
        return None

    def readData(self, inputFiles, genreparseddata, wikidata, prop, vocab_file, spine_word_vectors, spine_features, spine_dim):
        for num, inputFile in enumerate(inputFiles):
            if not inputFile:
                continue
            genreparseddata_ = genreparseddata[num]
            self.lang_full = inputFile.strip().split('/')[-2].split('-')[0][3:]
            f = inputFile.strip()
            data = pyconll.load_from_file(f"{f}")

            is_test = False
            if "test" in inputFile:
                is_test = True
            index = 0

            for sentence in data:
                text = sentence.text
                id2index = sentence._ids_to_indexes
                token_data, genre_token_data = None, None
                if self.args.use_wikid and genreparseddata_ and text in genreparseddata_:
                    genre_token_data = genreparseddata_[text]

                # Add the head-dependents
                dep_data_token = defaultdict(list)
                for token_num, token in enumerate(sentence):
                    dep_data_token[token.head].append(token.id)

                for token_num, token in enumerate(sentence):
                    token_id = token.id
                    if "-" in token_id or "." in token_id and not self.isValid(token.deprel):
                        continue
                    relation = token.deprel
                    pos = token.upos
                    feats = token.feats
                    lemma = token.lemma

                    if prop not in feats:
                        continue
                    label = self.getFeatureValue(prop, feats)


                    if token.head and token.head != "0" and pos and relation:
                        relation = relation.lower()
                        head_pos = sentence[token.head].upos
                        head_feats = sentence[token.head].feats
                        headrelation = sentence[token.head].deprel
                        headhead = sentence[token.head].head
                        head_lemma = sentence[token.head].lemma

                        if pos not in self.model_dictionary:
                            self.model_dictionary[pos] = len(self.model_dictionary)
                            self.model_data_case[pos] = defaultdict(lambda: 0)
                            self.feature_freq[pos] = defaultdict(lambda : 0)
                            self.examples_per_pos[pos] = defaultdict(lambda: 0)

                        self.feature_dictionary[label] += 1

                        if prop  in head_feats:
                            self.examples_per_pos[pos][token.lemma] += 1
                            headlabel = self.getFeatureValue(prop, head_feats)
                            feature = f'deppos_{pos}'
                            if not is_test:
                                self.feature_freq[pos][feature] += 1

                            feature = f'headpos_{head_pos}'
                            if not is_test:
                                self.feature_freq[pos][feature] += 1

                            if self.isValid(relation):
                                feature = f'deprel_{relation}'
                                if not is_test:
                                    self.feature_freq[pos][feature] += 1

                            if not self.args.only_triples: #adding dependents of the dep

                                for feat in feats: #self.required_features:
                                    if prop in feat or prop in self.remove_features:#skip the propery being modeled
                                        continue
                                    feature = f'depfeat_{feat}_'
                                    value = self.getFeatureValue(feat, feats)
                                    feature += f'{value}'
                                    if not is_test:
                                        self.feature_freq[pos][feature] += 1


                                if headrelation:
                                    headrelation = headrelation.lower()
                                    if self.isValid(headrelation): #headrelation != 'root' and headrelation != 'punct' and "unk" not in headrelation:
                                        feature = f'headrelrel_{headrelation.lower()}'
                                        if not is_test:
                                            self.feature_freq[pos][feature] += 1

                                for feat in head_feats:  # Adding features for dependent token (maybe more commonly occurring)
                                    if prop in feat or prop in self.remove_features:
                                        continue
                                    feature = f'headfeat_{head_pos}_{feat}_'
                                    value = self.getFeatureValue(feat, head_feats)
                                    feature += f'{value}'
                                    if not is_test:
                                        self.feature_freq[pos][feature] += 1

                                    feature = f'headfeat_{feat}_{value}'
                                    if not is_test:
                                        self.feature_freq[pos][feature] += 1

                                # Adding lemma of the head's head
                                if self.args.lexical:


                                    lemma = self.isValidLemma(lemma, pos)
                                    if lemma:
                                        feature = f'lemma_{lemma}'
                                        if not is_test:
                                            self.feature_freq[pos][feature] += 1

                                    head_lemma = self.isValidLemma(head_lemma, head_pos)
                                    if head_lemma:
                                        feature = f'headlemma_{head_lemma}'
                                        if not is_test:
                                            self.feature_freq[pos][feature] += 1

                                    # Add tokens in the neighborhood of 3
                                    neighboring_tokens_left = max(0, token_num - 3)
                                    neighboring_tokens_right = min(token_num + 3, len(sentence))
                                    for neighor in range(neighboring_tokens_left, neighboring_tokens_right):
                                        if neighor == token_num and neighor >= len(sentence):
                                            continue
                                        neighor_token = sentence[neighor]
                                        if neighor_token:
                                            lemma = self.isValidLemma(neighor_token.lemma, neighor_token.upos)
                                            if lemma:
                                                feature = f'neighborhood_{lemma}'
                                                if not is_test:
                                                    self.feature_freq[pos][feature] += 1

                                # get other dep tokens of the head
                                dep = dep_data_token.get(token.head, [])
                                for d in dep:
                                    if d == token.id:
                                        continue
                                    depdeprelation = sentence[d].deprel
                                    if depdeprelation:
                                        depdeprelation = depdeprelation.lower()
                                        if self.isValid(depdeprelation): #depdeprelation != 'root' and depdeprelation != 'punct' and 'unk' not in depdeprelation:
                                            feature = f'depheadrel_{depdeprelation}'
                                            if not is_test:
                                                self.feature_freq[pos][feature] += 1

                                    depdeppos = sentence[d].upos
                                    feature = f'depheadpos_{depdeppos}'
                                    if not is_test:
                                        self.feature_freq[pos][feature] += 1

                                if headhead and headhead != '0':
                                    headhead_feats = sentence[headhead].feats
                                    if prop in headhead_feats:
                                        headhead_value =   self.getFeatureValue(prop, headhead_feats)
                                        if headlabel == headhead_value:
                                            feature = f'headmatch_True'
                                            if not is_test:
                                                self.feature_freq[pos][feature] += 1

                            if self.args.use_wikid and genre_token_data:
                                if token_id in genre_token_data:
                                    qids_all = genre_token_data[token_id]
                                    features = self.getWikiDataGenreParse(qids_all, wikidata)
                                    for feature in features:
                                        if not is_test:
                                            feature = 'wiki_' + feature
                                            self.feature_freq[pos][feature] += 1

                                if token.head in genre_token_data:
                                    qids_all = genre_token_data[token.head]
                                    features = self.getWikiDataGenreParse(qids_all, wikidata)
                                    for feature in features:
                                        feature = 'wiki_head_' + feature
                                        if not is_test:
                                            self.feature_freq[pos][feature] += 1

                            if self.args.use_spine:
                                vector = self.getSpineFeatures(spine_word_vectors, token.form, spine_dim, token.lemma)
                                feature_names = spine_features[vector == 1]  # Get active features
                                for feature in feature_names:
                                    feature = f'spine_{feature}'
                                    if not is_test:
                                        self.feature_freq[pos][feature] += 1

                                if token.head and token.head != '0':
                                    vector = self.getSpineFeatures(spine_word_vectors, sentence[token.head].form, spine_dim, sentence[token.head].lemma)
                                    feature_names = spine_features[vector == 1]  # Get active features
                                    for feature in feature_names:
                                        feature = f'spinehead_{feature}'
                                        if not is_test:
                                            self.feature_freq[pos][feature] += 1

                            if label == headlabel: #If agreement between the head-dep
                                agreelabel = 1
                            else:
                                agreelabel = 0

                            self.model_data[pos] += 1
                            self.model_data_case[pos][agreelabel] += 1 #Label distribution per each POS tag

                            index += 1

        self.id2model = {v: k for k, v in self.model_dictionary.items()}
        self.feature_map_id2model = {}
        if self.args.transliterate:
            trn = Transliterator(source=self.args.transliterate, target='eng', build_lookup=True)
        else:
            trn = None
        with open(vocab_file, 'w') as fout:
            for model, items in self.feature_freq.items():
                self.feature_map[model] = {}
                for feature, freq in items.items():
                    if freq < 50:
                        continue
                    self.feature_map[model][feature] = len(self.feature_map[model])
                self.feature_map_id2model[model] = {v: k for k, v in self.feature_map[model].items()}
                fout.write(f'Model:{model}\n')
                for k, v in self.feature_map[model].items():
                    fout.write(f'Feature:{v}\t{k}\n')

                # labels = self.model_data_case[model]
                # fout.write(f'Data:{self.model_data[model]}\tLabels\t')
                # label_values = []
                fout.write(f'Labels:0-0,1-1\n')
                sorted_pos_examples = sorted(self.examples_per_pos[model].items(), key=lambda kv: kv[1], reverse=True)[:5]
                for (example, _) in sorted_pos_examples:
                    if trn:
                        fout.write(f'POS:{example} ({trn.transform(example)})\n')
                    else:
                         fout.write(f'POS:{example}\n')
                fout.write("\n")
                # fout.write(";".join(label_values) + "\n")

            fout.write('\n')

    def getFeatureValue(self, feat, feats):
        values = list(feats[feat])
        values.sort()
        value = "/".join(values)
        return value

    def getWikiDataParse(self, babel_ids, wikidata):
        features = []
        for babel_id in babel_ids:
            if babel_id in wikidata:
                wikifeatures = wikidata[babel_id]
                for wiki, _ in wikifeatures.items():
                    feature = f'{wiki}'
                    features.append(feature)
        return features

    def getWikiDataGenreParse(self, qids_all, wikidata):
        features = []
        for qids in qids_all: #qids = [Q12, Q23, ...]
            for qid in qids.split(","):
                if qid in wikidata:
                    wikifeatures = wikidata[qid]
                    for wiki, _ in wikifeatures.items():
                        feature = f'wiki_{wiki}'
                        features.append(feature)
                    break
        return features

    def getSpineFeatures(self, spine_word_vectors, word, dim, lemma):
        vector = [0 for _ in range(dim)]
        if word.lower() in spine_word_vectors:
            vector = spine_word_vectors[word.lower()]
            if lemma not in spine_word_vectors:
                spine_word_vectors[lemma] = vector
        elif lemma in spine_word_vectors:
            vector = spine_word_vectors[lemma]
        return np.array(vector)

    def isValid(self,relation):
        if not relation:
            return False
        found = False

        relation = relation.lower()
        if len(self.required_relations) > 0:  # Restrict analysis to relations
            for rel in self.required_relations:
                if rel in relation:
                    found = True
        else:
            found = True
        return found

    def addFeature(self, model, feature, feature_array, label, value=1):
        feature_id = self.feature_map[model].get(feature, -1)
        if feature_id >= 0:
            feature_array[feature_id] = value
            #self.model_feature_label[model][label][feature] += 1

    def getAssignmentFeatures(self, inputFile,
                                  genreparseddata,
                                  wikidata, prop, model,
                              spine_word_vectors, spine_features, spine_dim, filename):

        # if model not in self.model_feature_label:
        #     self.model_feature_label[model] = {}
        f = inputFile.strip()
        data = pyconll.load_from_file(f"{f}")
        all_features = []
        columns = []
        output_labels = defaultdict(lambda : 0)
        num_of_tokens_syntax = 0
        num_of_tokens_wikigenre = 0
        num_of_tokens_spine = 0
        num_of_tokens_lexical = 0

        index = [i for i in range(len(data))]
        os.system(f'rm -rf {filename}')
        #Get column names
        for i in range(len(self.feature_map[model])):
            feature_name = self.feature_map_id2model[model][i]
            columns.append(feature_name)
        columns.append('label')
        columns.append('sent_num')
        columns.append('token_num')
        num = 0
        for sentence_num in index:
            sentence = data[sentence_num]
            text = sentence.text
            token_data, genre_token_data = None, None
            if self.args.use_wikid and genreparseddata and text in genreparseddata:
                genre_token_data = genreparseddata[text]

            # Add the head-dependents
            dep_data_token = defaultdict(list)
            for token_num, token in enumerate(sentence):
                dep_data_token[token.head].append(token.id)

            for token_num, token in enumerate(sentence):
                token_id = token.id
                if "-" in token_id or "." in token_id and not self.isValid(token.deprel):
                    continue
                relation = token.deprel.lower()
                pos = token.upos
                feats = token.feats
                lemma = token.lemma

                if pos != model:
                    continue

                if prop not in feats:
                    continue
                label = self.getFeatureValue(prop, feats)
                feature_array = np.zeros((len(self.feature_map[model]),), dtype=int)

                if token.head and token.head != "0" and pos and relation:
                    head_pos = sentence[token.head].upos
                    head_feats = sentence[token.head].feats
                    headrelation = sentence[token.head].deprel
                    headhead = sentence[token.head].head
                    head_lemma = sentence[token.head].lemma


                    if prop in head_feats:
                        headlabel = self.getFeatureValue(prop, head_feats)
                        if label == headlabel:  # If agreement between the head-dep
                            agreelabel = 1
                        else:
                            agreelabel = 0



                        feature = f'deppos_{pos}'
                        self.addFeature(model, feature, feature_array, agreelabel)


                        feature = f'headpos_{head_pos}'
                        self.addFeature(model, feature, feature_array, agreelabel)

                        feature = f'deprel_{relation}'
                        self.addFeature(model, feature, feature_array, agreelabel)

                        if not self.args.only_triples:  # adding dependents of the dep
                            for feat in feats:  # self.required_features:
                                if prop in feat:  # skip the propery being modeled
                                    continue
                                feature = f'depfeat_{feat}_'
                                value = self.getFeatureValue(feat, feats)
                                feature += f'{value}'
                                self.addFeature(model, feature, feature_array, agreelabel)

                            if headrelation and headrelation != 'root' and headrelation != 'punct':
                                feature = f'headrelrel_{headrelation.lower()}'
                                self.addFeature(model, feature, feature_array, agreelabel)

                            for feat in head_feats:  # Adding features for dependent token (maybe more commonly occurring)
                                if prop in feat:
                                    continue
                                feature = f'headfeat_{head_pos}_{feat}_'
                                value = self.getFeatureValue(feat, head_feats)
                                feature += f'{value}'
                                self.addFeature(model, feature, feature_array, agreelabel)

                                feature = f'headfeat_{feat}_{value}'
                                self.addFeature(model, feature, feature_array, agreelabel)

                            if headhead and headhead != '0':
                                headhead_feats = sentence[headhead].feats
                                headheadhead = sentence[headhead].head
                                headhead_value = None
                                if prop in headhead_feats:
                                    headhead_value = self.getFeatureValue(prop, headhead_feats)
                                    if headlabel == headhead_value:
                                        feature = f'headmatch_True'
                                        self.addFeature(model, feature, feature_array, agreelabel)

                            # Adding lemma of the head's head
                            if self.args.lexical:
                                # headheadhead = sentence[token.head].head
                                # if headheadhead != '0':
                                #     headheadheadlemma = self.isValidLemma(sentence[headheadhead].lemma,
                                #                                           sentence[headheadhead].upos)
                                #     if headheadheadlemma:
                                #         feature = f'headheadlemma_{headheadheadlemma}'
                                #         self.addFeature(feature, feature_array)

                                lemma = self.isValidLemma(lemma, pos)
                                if lemma:
                                    feature = f'lemma_{lemma}'
                                    self.addFeature(model, feature, feature_array, agreelabel)

                                head_lemma = self.isValidLemma(head_lemma, head_pos)
                                if head_lemma:
                                    feature = f'headlemma_{head_lemma}'
                                    self.addFeature(model, feature, feature_array, agreelabel)

                                # Add tokens in the neighborhood of 3
                                neighboring_tokens_left = max(0, token_num - 3)
                                neighboring_tokens_right = min(token_num + 3, len(sentence))
                                for neighor in range(neighboring_tokens_left, neighboring_tokens_right):
                                    if neighor == token_num and neighor >= len(sentence):
                                        continue
                                    neighor_token = sentence[neighor]
                                    if neighor_token:
                                        lemma = self.isValidLemma(neighor_token.lemma, neighor_token.upos)
                                        if lemma:
                                            feature = f'neighborhood_{lemma}'
                                            self.addFeature(model, feature, feature_array, agreelabel)

                            # get other dep tokens of the head
                            dep = dep_data_token.get(token.head, [])
                            for d in dep:
                                if d == token.id:
                                    continue
                                depdeprelation = sentence[d].deprel
                                if depdeprelation and depdeprelation != 'punct':
                                    feature = f'depheadrel_{depdeprelation}'
                                    self.addFeature(model, feature, feature_array, agreelabel)

                                depdeppos = sentence[d].upos
                                feature = f'depheadpos_{depdeppos}'
                                self.addFeature(model, feature, feature_array, agreelabel)


                        if self.args.use_wikid and genre_token_data:
                            isWiki = False
                            if token_id in genre_token_data:
                                qids_all = genre_token_data[token_id]
                                features = self.getWikiDataGenreParse(qids_all, wikidata)
                                for feature in features:
                                    feature = 'wiki_' + feature
                                    self.addFeature(model, feature, feature_array, agreelabel)
                                    isWiki = True
                                if isWiki:
                                    num_of_tokens_wikigenre += 1
                            isWiki = False
                            if token.head in genre_token_data:
                                qids_all = genre_token_data[token.head]
                                features = self.getWikiDataGenreParse(qids_all, wikidata)
                                for feature in features:
                                    feature = 'wiki_head_' + feature
                                    self.addFeature(model, feature, feature_array, agreelabel)
                                    isWiki = True

                                if isWiki:
                                    num_of_tokens_wikigenre += 1

                        if self.args.use_spine:
                            isspine = False
                            vector = self.getSpineFeatures(spine_word_vectors, token.form, spine_dim, token.lemma)
                            feature_names = spine_features[vector == 1]  # Get active features
                            vector_filtered = vector[vector > 0]
                            for feature, value in zip(feature_names, vector_filtered):
                                feature = f'spine_{feature}'
                                self.addFeature(model, feature, feature_array, agreelabel, value=value)
                                isspine = True

                            if isspine:
                                num_of_tokens_spine +=  1

                            isspine = False
                            if token.head and token.head != '0':
                                vector = self.getSpineFeatures(spine_word_vectors, sentence[token.head].form, spine_dim,
                                                               sentence[token.head].lemma)
                                feature_names = spine_features[vector > 0]  # Get active features
                                vector_filtered = vector[vector > 0]
                                for feature, value in zip(feature_names, vector_filtered):
                                    feature = f'spinehead_{feature}'
                                    self.addFeature(model, feature, feature_array, agreelabel, value=value)



                        one_feature = np.concatenate(
                            (feature_array, int(agreelabel), sentence_num, token.id), axis=None)

                        # get feature indexes which are not 0
                        feature_indexes = list(np.nonzero(feature_array)[0])
                        nonzero_values = []
                        for index in feature_indexes:
                            nonzero_values.append(f'{index}:{feature_array[index]}')
                        feature_indexes = [str(agreelabel)] + nonzero_values  # have 1 index file
                        assert len(one_feature) == len(columns)
                        #all_features.append(one_feature)
                        output_labels[agreelabel] += 1
                        num += 1
                        with open(filename, 'a') as fout:
                            fout.write("\t".join(feature_indexes) + "\n")
                        one_feature = np.concatenate((sentence_num, token.id), axis=None)
                        # get feature indexes which are not 0
                        df = pd.DataFrame([one_feature], columns=['sent_num', 'token_num'])
                        infofilename = filename.replace("libsvm", "")
                        df.to_csv(infofilename, mode='a', header=not os.path.exists(infofilename))

        print(
            f'Syntax: {num_of_tokens_syntax}  Wikigenre: {num_of_tokens_wikigenre}, Lexical: {num_of_tokens_lexical}, Spine:{num_of_tokens_spine} columns: {len(columns)}')
        #random.shuffle(all_features)
        return all_features, columns, output_labels

    def getFeaturesForToken(self, sent, token, allowed_features,
                    genreparseddata,
                    wikidata,
                    spine_word_vectors,
                    spine_features,
                    spine_dim,prop):

        text = sent.text
        id2index = sent._ids_to_indexes
        token_id = token.id
        token_data, genre_token_data = None, None
        if self.args.use_wikid and genreparseddata and text in genreparseddata:
            genre_token_data = genreparseddata[text]

        dep_data_token = defaultdict(list)
        for token_num, token_ in enumerate(sent):
            dep_data_token[token_.head].append(token_.id)

        relation = token.deprel.lower()
        pos = token.upos
        feats = token.feats
        lemma = token.lemma
        head_pos = sent[token.head].upos
        head_feats = sent[token.head].feats
        head_lemma = sent[token.head].lemma
        head_relation = sent[token.head].deprel
        headhead = sent[token.head].head

        feature_array = {}

        feature = f'headpos_{head_pos}'
        if feature in allowed_features:
            feature_array[feature] = 1

        feature = f'deppos_{pos}'
        if feature in allowed_features:
            feature_array[feature] = 1

        feature = f'deprel_{relation}'
        if feature in allowed_features:
            feature_array[feature] = 1

        if not self.args.only_triples:
            headlabel = self.getFeatureValue(prop, head_feats)

            for feat in feats:  # self.required_features:
                feature = f'depfeat_{feat}_'
                value = self.getFeatureValue(feat, feats)
                feature += f'{value}'
                if feature in allowed_features:
                    feature_array[feature] = 1

            if head_relation and head_relation != 'root' and head_relation != 'punct':
                feature = f'headrelrel_{head_relation.lower()}'
                if feature in allowed_features:
                    feature_array[feature] = 1

            for feat in head_feats:  # Adding features for dependent token (maybe more commonly occurring)
                feature = f'headfeat_{head_pos}_{feat}_'
                value = self.getFeatureValue(feat, head_feats)
                feature += f'{value}'
                if feature in allowed_features:
                    feature_array[feature] = 1

                feature = f'headfeat_{feat}_{value}'
                if feature in allowed_features:
                    feature_array[feature] = 1

            if headhead and headhead != '0':
                headhead_feats = sent[headhead].feats
                if prop in headhead_feats:
                    headhead_value = self.getFeatureValue(prop, headhead_feats)
                    if headlabel == headhead_value:
                        feature = f'headmatch_True'
                        if feature in allowed_features:
                            feature_array[feature] = 1
            # Adding lemma of the head's head
            if self.args.lexical:
                lemma = self.isValidLemma(lemma, pos)
                if lemma:
                    feature = f'lemma_{lemma}'
                    if feature in allowed_features:
                        feature_array[feature] = 1

                head_lemma = self.isValidLemma(head_lemma, head_pos)
                if head_lemma:
                    feature = f'headlemma_{head_lemma}'
                    if feature in allowed_features:
                        feature_array[feature] = 1

                # Add tokens in the neighborhood of 3
                neighboring_tokens_left = max(0, token_num - 3)
                neighboring_tokens_right = min(token_num + 3, len(sent))
                for neighor in range(neighboring_tokens_left, neighboring_tokens_right):
                    if neighor == token_num and neighor >= len(sent):
                        continue
                    neighor_token = sent[neighor]
                    if neighor_token:
                        lemma = self.isValidLemma(neighor_token.lemma, neighor_token.upos)
                        if lemma:
                            feature = f'neighborhood_{lemma}'
                            if feature in allowed_features:
                                feature_array[feature] = 1

            dep = dep_data_token.get(token.head, [])
            for d in dep:
                if d == token.id:
                    continue
                depdeprelation = sent[d].deprel
                if depdeprelation:
                    # get the deprel of other deps of the head
                    feature = f'depheadrel_{depdeprelation}'
                    if feature in allowed_features:
                        feature_array[feature] = 1

                depdeppos = sent[d].upos
                feature = f'depheadpos_{depdeppos}'
                if feature in allowed_features:
                    feature_array[feature] = 1

                depdeplemma = self.isValidLemma(sent[d].lemma, sent[d].upos)
                if depdeplemma:
                    feature = f'depheadlemma_{depdeplemma}'
                    if feature in allowed_features:
                        feature_array[feature] = 1

        if self.args.use_wikid:
            # Get the wikidata features
            if genre_token_data:
                isWiki = False
                if token_id in genre_token_data:
                    qids_all = genre_token_data[token_id]
                    features = self.getWikiDataGenreParse(qids_all, wikidata)
                    for feature in features:
                        feature = 'wiki_' + feature
                        if feature in allowed_features:
                            feature_array[feature] = 1

                if token.head in genre_token_data:
                    head_id = token.head
                    qids_all = genre_token_data[head_id]
                    features = self.getWikiDataGenreParse(qids_all, wikidata)
                    for feature in features:
                        feature = 'wiki_head_' + feature
                        if feature in allowed_features:
                            feature_array[feature] = 1

        if self.args.use_spine:
            vector = self.getSpineFeatures(spine_word_vectors, token.form, spine_dim, token.lemma)
            feature_names = spine_features[vector > 0]  # Get active features
            vector_filtered = vector[vector > 0]
            for feature, value in zip(feature_names, vector_filtered):
                feature = f'spine_{feature}'
                if feature in allowed_features:
                    feature_array[feature] = value

            if token.head and token.head != '0':
                vector = self.getSpineFeatures(spine_word_vectors, sent[token.head].form, spine_dim,
                                               sent[token.head].lemma)
                feature_names = spine_features[vector > 0]  # Get active features
                vector_filtered = vector[vector > 0]
                for feature, value in zip(feature_names, vector_filtered):
                    feature = f'spinehead_{feature}'
                    if feature in allowed_features:
                        feature_array[feature] = value



        return feature_array

    def getHistogram(self, filename, input_path, feature):
        f = input_path.strip()
        data = pyconll.load_from_file(f"{f}")
        self.feature_tokens, self.feature_forms = {}, {}
        self.feature_tokens[feature] = defaultdict(lambda: 0)
        self.feature_forms = {}
        self.feature_forms_num = {}
        tokens, feature_values, pos_values= [], [], []
        self.lemma, self.lemmaGroups, self.lemma_freq, self.lemma_inflection = {}, defaultdict(set), {}, {}
        pos_barplots = {}
        features_set, pos_count = set(), defaultdict(lambda : 0)
        label_value = defaultdict(lambda : 0)

        for sentence in data:
            for token in sentence:
                if token.form == None or "-" in token.id:
                    continue

                token_id = token.id
                relation = token.deprel
                pos = token.upos
                if pos == None:
                    pos = 'None'
                feats = token.feats
                lemma = token.lemma

                tokens.append(token.form)
                pos_values.append(pos)

                pos_count[pos] += 1
                self.lemma[token.form.lower()] = lemma
                self.lemmaGroups[lemma].add(token.form.lower())
                if pos not in self.feature_forms_num:
                    self.feature_forms_num[pos] = {}
                    pos_barplots[pos] = defaultdict(lambda : 0)

                if pos not in self.lemma_inflection:
                    self.lemma_freq[pos] = defaultdict(lambda: 0)
                    self.lemma_inflection[pos] = {}

                if lemma:
                    self.lemma_freq[pos][lemma.lower()] += 1
                if lemma and lemma.lower() not in self.lemma_inflection[pos]:
                    self.lemma_inflection[pos][lemma.lower()] = {}
                # Aggregae morphology properties of required-properties - feature
                morphology_props = set(self.args.features) - set([feature])
                morphology_prop_values = []
                for morphology_prop in morphology_props:
                    if morphology_prop in feats:
                        morphology_prop_values.append(",".join(feats[morphology_prop]))
                morphology_prop_values.sort()
                inflection = ";".join(morphology_prop_values)
                if lemma and inflection not in self.lemma_inflection[pos][lemma.lower()]:
                    self.lemma_inflection[pos][lemma.lower()][inflection] = {}
                if feature in feats:
                    values = list(feats[feature])
                    values.sort()
                    feature_values.append("/".join(values))
                    label_value["/".join(values)] += 1
                    #for feat in values:
                else:
                    values = ['NA']
                    feature_values.append("NA")
                    label_value['NA'] += 1
                feat = "/".join(values)
                features_set.add(feat)
                pos_barplots[pos][feat] += 1
                if feat not in self.feature_forms_num[pos]:
                    self.feature_forms_num[pos][feat] = defaultdict(lambda : 0)

                self.feature_forms_num[pos][feat][token.form.lower()] += 1
                if lemma:
                    self.lemma_inflection[pos][lemma.lower()][inflection][feat] = token.form.lower()

        #sort the pos by count
        sorted_pos = sorted(pos_count.items(), key= lambda kv: kv[1], reverse=True)
        pos_to_id, pos_order = {}, []
        for (pos, _) in sorted_pos:
            pos_to_id[pos] = len(pos_to_id)
            pos_order.append(pos)

        #Stacked histogram
        #sns.set()
        fig, ax = plt.subplots()
        bars_num = np.zeros((len(features_set), len(pos_barplots)))
        x_axis = []
        feat_to_id, id_to_feat = get_vocab_from_set(features_set)

        for pos in pos_order:
            feats = pos_barplots[pos]
            x_axis.append(pos)
            pos_id = pos_to_id[pos]
            for feat, num in feats.items():
                feat_id = feat_to_id[feat]
                bars_num[feat_id][pos_id] = num

        r = [i for i in range(len(pos_to_id))]
        handles, color = [], ['steelblue', 'orange', 'olivedrab', 'peru', 'seagreen', 'chocolate',
                              'tan', 'lightseagreen', 'green', 'teal','tomato','lightgreen','yellow','lightblue','azure','red',
                              'aqua', 'darkgreen', 'tomato', 'firebrick', 'khaki', 'gold', 'powderblue',  'navy', 'plum' ]
        bars = np.zeros((len(pos_barplots)))
        handle_map = {}
        for barnum in range(len(features_set)):
            plt.bar(r, bars_num[barnum], bottom=bars, color=color[barnum], edgecolor='white', width=1)
            handle_map[id_to_feat[barnum]]= mpatches.Patch(color=color[barnum], label=id_to_feat[barnum])
            bars += bars_num[barnum]

        #Sort legend by frequency
        sorted_labels = sorted(label_value.items(), key=lambda kv:kv[1], reverse=True)
        for (label, _) in sorted_labels:
            handles.append(handle_map[label])

        #handles.reverse()
        # Custom X axis
        plt.xticks(r, x_axis,rotation=45, fontsize=9)
        #plt.xlabel("pos")
        plt.ylabel("Number of Tokens")
        plt.legend(handles=handles)

        right_side = ax.spines["right"]
        right_side.set_visible(False)

        top_side = ax.spines["top"]
        top_side.set_visible(False)

        plt.savefig(f"{filename}" + "/pos.png", transparent=True)
        plt.close()
        return pos_order

    def example_web_print(self, ex, outp2, data):
        try:
            # print('\t\t',data[ex[0]].text)
            sentid = int(ex[0])
            tokid = str(ex[1])
            active  = ex[2]
            req_head_head = False
            for feature in active:
                info = feature.split("_")
                if len(info) > 2:
                    if feature.startswith("agree") or feature.startswith('headrelrel'):
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
                        outp2.write(token.conll() + "\n")
                    else:
                        temp = token.conll().split('\t')
                        temp2 = '\t'.join(temp[:6])
                        outp2.write(f"{temp2}\t0\t_\t_\t_\n")
                elif token.id == data[sentid][headid].head and req_head_head:
                    temp = token.conll().split('\t')
                    temp2 = '\t'.join(temp[:6])
                    outp2.write(f"{temp2}\t0\t_\t_\t_\n")
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

    def addFeaturesHead(self, token):
        features = []

        for feat in self.required_features:  # Adding features for dependent token (maybe more commonly occurring)
            feats = token.feats
            if feat in feats:
                feature = f'{feat}_'
                value = self.getFeatureValue(feat, feats)
                feature += f'{value}'
                features.append(feature)
        return features


