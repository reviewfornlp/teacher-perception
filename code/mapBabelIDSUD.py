import pyconll, argparse
from collections import defaultdict
from nltk.corpus import wordnet as wn
import os

parser = argparse.ArgumentParser()
parser.add_argument("--en_input", type=str, default="en_sam-sud-train.conllu")
parser.add_argument("--mr_input", type=str, default="mr_sam-sud-train.conllu")
parser.add_argument("--alignment", type=str, default="train.pred")
parser.add_argument("--wsd_output_pred", type=str, default="wsd-en-mr.predictions.txt")
parser.add_argument("--wsd", type=str, default="babelid_synset.txt")
parser.add_argument("--output", type=str, default="vocab_wordnet.txt")
args = parser.parse_args()

def parseAlignment(info):
    src_tgt_alignments = defaultdict(list)
    tgt_src_alignments = defaultdict(list)
    for align in info:
        s = int(align.split('-')[0])
        t = int(align.split('-')[1])
        src_tgt_alignments[s].append(t)
        tgt_src_alignments[t].append(s)
    return src_tgt_alignments, tgt_src_alignments

if __name__ == "__main__":
    wsd_info = {}
    with open(args.wsd, 'r') as fin:
        for line in fin.readlines():
            if line == "" or line == "\n":
                continue
            if line.startswith('babel_id'):
                continue
            info = line.strip().split("\t")
            #print(info)
            #if info[1] == 'relation':
            #    continue
            babelid, wordnetid = info[0], info[-1]
            if babelid not in wsd_info and wordnetid.startswith("wn:"):
                wsd_info[babelid] = wordnetid

    wordnetpaths = defaultdict(set)
    word_pairs = {}
    words = defaultdict(lambda: 0)
    wordnetpaths_types = defaultdict(lambda : 0)
    examples = {}
    depths = {}

    en_input_file, mr_input_file, alignment_input_file = args.en_input, args.mr_input, args.alignment
    with open(alignment_input_file, 'r') as fin:
        alignmentdata = fin.readlines()

    if not os.path.exists(en_input_file) or not os.path.exists(mr_input_file):
        print(f'Input files: {en_input_file} or {mr_input_file} not present!')
        exit(-1)

    with open(args.wsd_output_pred, 'r') as fin:
        wsd_pred_outputs = {}
        for line in fin.readlines():
            if '<unk>' in line:
                continue
            info = line.strip().split()
            wsd_pred_outputs[info[0]] = info[1]
    data = pyconll.load_from_file(en_input_file)
    mr_data = pyconll.load_from_file(mr_input_file)


    for sent_tokenid, babelid in wsd_pred_outputs.items():
        if babelid not in wsd_info:
            continue
        sent = int(sent_tokenid.split(".")[0])
        token_id = sent_tokenid.split(".")[1]

        alignments_src, alignments_tgt = parseAlignment(
            alignmentdata[sent].strip().replace("p", "-").split())  # en->mr
        en_sent, mr_sent = data[sent], mr_data[sent]

        id2index = en_sent._ids_to_indexes

        en_token_index = id2index[token_id]
        aligned_mr_token_indexes = alignments_src.get(en_token_index, [])

        wordnetid = wsd_info[babelid].split("wn:")[1]
        synset = wn.synset_from_pos_and_offset(wordnetid[-1], int(wordnetid[:-1]))
        all_paths = synset.hypernym_paths()

        en_word = en_sent[en_token_index]
        mr_words = []
        mr_pos_words = []
        for mr_index in aligned_mr_token_indexes:
            mr_word = mr_sent[mr_index]
            if mr_word.lemma:
                mr_words.append(mr_word.lemma)
            else:
                mr_words.append(mr_word.form)
            mr_pos_words.append(mr_word.upos)

        if en_word.lemma:
            en_word_lemma = en_word.lemma.lower()
        else:
            en_word_lemma = en_word.form.lower()
        if en_word.upos not in mr_pos_words:
            continue

        if en_word_lemma not in word_pairs:
            word_pairs[en_word_lemma] = defaultdict(lambda: 0)
            examples[en_word_lemma] = defaultdict(list)

        mr_words_text = " ".join(mr_words)
        word_pairs[en_word_lemma][mr_words_text] += 1
        for syn in all_paths[0]:
            depth = all_paths[0].index(syn)
            synname = syn._name
            wordnetpaths[synname].add(en_word_lemma)
            depths[synname] = depth

        examples[en_word_lemma][mr_words_text].append(
            (sent, token_id, en_input_file, mr_input_file, alignment_input_file))

    for sent_num, sent in enumerate(data):
        alignments_src, alignments_tgt = parseAlignment(
            alignmentdata[sent_num].strip().replace("p", "-").split())  # en->mr
        en_sent, mr_sent = data[sent_num], mr_data[sent_num]

        id2index = en_sent._ids_to_indexes
        for en_token_index, token in enumerate(sent):
            aligned_mr_token_indexes = alignments_src.get(en_token_index, [])

            en_word = en_sent[en_token_index]
            mr_words = []
            for mr_index in aligned_mr_token_indexes:
                mr_word = mr_sent[mr_index]
                if mr_word.lemma:
                    mr_words.append(mr_word.lemma)
                else:
                    mr_words.append(mr_word.form)

            if en_word.lemma:
                en_word_lemma = en_word.lemma.lower()
            else:
                en_word_lemma = en_word.form.lower()

            if en_word_lemma not in word_pairs:
                word_pairs[en_word_lemma] = defaultdict(lambda: 0)
                examples[en_word_lemma] = defaultdict(list)

            mr_words_text = " ".join(mr_words)
            word_pairs[en_word_lemma][mr_words_text] += 1
            examples[en_word_lemma][mr_words_text].append(
                (sent_num, token.id, en_input_file, mr_input_file, alignment_input_file))

    '''
    Code to read multiple files if a single file is large, change the below code depending on your file naming format.
    Below code works when train file is named as for English sentences (train.en.aa.conllu), its corresponding Marathi sentences (train.mr.aa.conllu), and its alignment (train.en.mr.pred.aa)   
    for [path, dir, files] in os.walk(args.en_input):
        for file in files:
            if "train.en" not in file:
                continue
            prefix = file.split("train.en.")[1].split(".conllu")[0]
            en_input_file, mr_input_file, alignment_input_file, wsd_output_pred_path = \
                f'{args.en_input}/train.en.{prefix}.conllu', f'{args.mr_input}/train.kn.sud.{prefix}.conllu', f'{args.alignment}/train.en.kn.pred.{prefix}', f'{args.wsd_output_pred}/wsd_{prefix}.predictions.txt'

            if os.path.exists(wsd_output_pred_path):
                with open(wsd_output_pred_path, 'r') as fin:
                    wsd_pred_outputs = {}
                    for line in fin.readlines():
                        if '<unk>' in line:
                            continue
                        info = line.strip().split()
                        wsd_pred_outputs[info[0]] = info[1]

            with open(alignment_input_file, 'r') as fin:
                alignmentdata = fin.readlines()

            if not os.path.exists(en_input_file) or not os.path.exists(mr_input_file):
                continue
            print('Processing: ', en_input_file, mr_input_file)
      
            data = pyconll.load_from_file(en_input_file)
            mr_data = pyconll.load_from_file(mr_input_file)

            if os.path.exists(wsd_output_pred_path):
                for sent_tokenid, babelid in wsd_pred_outputs.items():
                    if babelid not in wsd_info:
                        continue
                    sent = int(sent_tokenid.split(".")[0])
                    token_id = sent_tokenid.split(".")[1]

                    alignments_src, alignments_tgt = parseAlignment(alignmentdata[sent].strip().replace("p", "-").split()) #en->mr
                    en_sent, mr_sent = data[sent], mr_data[sent]

                    id2index = en_sent._ids_to_indexes

                    en_token_index = id2index[token_id]
                    aligned_mr_token_indexes = alignments_src.get(en_token_index, [])

                    wordnetid = wsd_info[babelid].split("wn:")[1]
                    synset = wn.synset_from_pos_and_offset(wordnetid[-1], int(wordnetid[:-1]))
                    all_paths = synset.hypernym_paths()


                    en_word = en_sent[en_token_index]
                    mr_words = []
                    mr_pos_words = []
                    for mr_index in aligned_mr_token_indexes:
                        mr_word = mr_sent[mr_index]
                        if mr_word.lemma:
                            mr_words.append(mr_word.lemma)
                        else:
                            mr_words.append(mr_word.form)
                        mr_pos_words.append(mr_word.upos)

                    if en_word.lemma:
                        en_word_lemma = en_word.lemma.lower()
                    else:
                        en_word_lemma = en_word.form.lower()
                    if en_word.upos not in mr_pos_words:
                        continue

                    if en_word_lemma not in word_pairs:
                        word_pairs[en_word_lemma] = defaultdict(lambda : 0)
                        examples[en_word_lemma] = defaultdict(list)

                    mr_words_text = " ".join(mr_words)
                    word_pairs[en_word_lemma][mr_words_text] += 1
                    for syn in all_paths[0]:
                        depth = all_paths[0].index(syn)
                        synname =syn._name
                        wordnetpaths[synname].add(en_word_lemma)
                        depths[synname] = depth

                    examples[en_word_lemma][mr_words_text].append((sent, token_id, en_input_file, mr_input_file, alignment_input_file ))

            for sent_num, sent in enumerate(data):
                alignments_src, alignments_tgt = parseAlignment(
                    alignmentdata[sent_num].strip().replace("p", "-").split())  # en->mr
                en_sent, mr_sent = data[sent_num], mr_data[sent_num]

                id2index = en_sent._ids_to_indexes
                for en_token_index, token in enumerate(sent):
                    aligned_mr_token_indexes = alignments_src.get(en_token_index, [])

                    en_word = en_sent[en_token_index]
                    mr_words = []
                    for mr_index in aligned_mr_token_indexes:
                        mr_word = mr_sent[mr_index]
                        if mr_word.lemma:
                            mr_words.append(mr_word.lemma)
                        else:
                            mr_words.append(mr_word.form)

                    if en_word.lemma:
                        en_word_lemma = en_word.lemma.lower()
                    else:
                        en_word_lemma = en_word.form.lower()

                    if en_word_lemma not in word_pairs:
                        word_pairs[en_word_lemma] = defaultdict(lambda: 0)
                        examples[en_word_lemma] = defaultdict(list)

                    mr_words_text = " ".join(mr_words)
                    word_pairs[en_word_lemma][mr_words_text] += 1
                    examples[en_word_lemma][mr_words_text].append(
                        (sent_num, token.id, en_input_file, mr_input_file, alignment_input_file))
    '''
    with open(args.output, 'w') as fout:
        for synname, en_word_lemmas in wordnetpaths.items():
            en_word_lemmas = list(en_word_lemmas)
            depth = depths[synname]
            fout.write(f'wsd:{synname}\tdepth:{depth}\t{",".join(en_word_lemmas)}\n')

        for en_word_lemma, mr_words in word_pairs.items():
            mr_words_string = []
            for mr_word, value in mr_words.items():
                mr_words_string.append(f'{mr_word}={value}')

            fout.write(f'en:{en_word_lemma}\t{";".join(mr_words_string)}\n')

    with open(args.output + ".examples", 'w') as fout:
        for en_word_lemma, mr_words in examples.items():
            mr_words_string = []
            sorted_mrwords = sorted(word_pairs[en_word_lemma].items(), key=lambda kv: kv[1], reverse=True)
            for (mr_word, _) in sorted_mrwords:
                for (sent, id, en_input_file, mr_input_file, alignment_input_file) in examples[en_word_lemma][mr_word][:10]:
                    fout.write(f'{en_word_lemma}\t{mr_word}\t{sent}\t{id}\t{en_input_file}\t{mr_input_file}\t{alignment_input_file}\n')



