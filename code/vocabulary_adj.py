from nltk.corpus import wordnet as wn
import argparse
from collections import defaultdict
from indictrans import Transliterator
import os
import pyconll
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--en_input", type=str, default="EN/en_sam-sud-dev.conllu")
parser.add_argument("--mr_input", type=str, default="MR/mr_sam-sud-dev.conllu")
parser.add_argument("--alignment", type=str, default="alignment/dev.pred")
parser.add_argument("--input", type=str)
parser.add_argument("--folder_name", type=str, default='./website/')
parser.add_argument("--lang", type=str, default="kn_en")
parser.add_argument("--transliterate", type=str, default="kan")
parser.add_argument("--en_adj", type=str, default="en_adj.txt")
args = parser.parse_args()

with open(f"{args.folder_name}/header.html") as inp:
    ORIG_HEADER = inp.readlines()
ORIG_HEADER = ''.join(ORIG_HEADER)

with open(f"{args.folder_name}/footer.html") as inp:
    FOOTER = inp.readlines()
FOOTER = ''.join(FOOTER)

input_text = f'<input type="text" id="myInput" class="myInput" style="width=155%" onkeyup="myFunction()" placeholder="Search for a word (e.g. rice)" title="word">\n'
script_text = '\n<script> \n function myFunction() {  var input, filter, table, tr, td, i, txtValue;\n' \
                              'input = document.getElementById("myInput");\n' \
                              'filter = input.value.toLowerCase();\n' \
                              'table = document.getElementById("myTable");\n' \
                              'tr = table.getElementsByTagName("tr");\n' \
                              'for (i = 0; i < tr.length; i++) {\n' \
                              'var colcount = tr[i].getElementsByTagName("td").length;\n' \
                              'td = tr[i].getElementsByTagName("td")[2];\n' \
                              'if (td) {\n' \
                              'txtValue = td.textContent || td.innerText;\n' \
                              'if (txtValue.toLowerCase().indexOf(filter) > -1) {\n' \
                              'tr[i].style.display = "";\n' \
                              '} else { \n' \
                              'tr[i].style.display = "none"; \n }}}} \n ' \
                              '</script>'

def parseAlignment(info):
    src_tgt_alignments = defaultdict(list)
    tgt_src_alignments = defaultdict(list)
    for align in info:
        s = int(align.split('-')[0])
        t = int(align.split('-')[1])
        src_tgt_alignments[s].append(t)
        tgt_src_alignments[t].append(s)
    return src_tgt_alignments, tgt_src_alignments

def outputExamplesWebPage(outp2, eng_sent, tgt_sent, tokid, alignment_sent):
    s2t, t2s = utils.parseAlignment(alignment_sent.strip().replace("p", "-").split())
    outp2.write('<pre><code class="language-conllu">\n')

    en_token_index = eng_sent._ids_to_indexes[tokid]

    aligned_mr_token_indexes = s2t.get(en_token_index, [])
    for token_num, token in enumerate(tgt_sent):
        transliterated = trn.transform(token.form)
        if token_num in aligned_mr_token_indexes:
            temp = token.conll().split('\t')
            temp[1] = "***" + temp[1] + '(' + transliterated + ')' + "***"
            temp2 = '\t'.join(temp)
            outp2.write(temp2 + "\n")

        elif '-' not in token.id:
            temp = token.conll().split('\t')
            temp[1] = temp[1] + '(' + transliterated + ')'

            temp[6], temp[7], temp[8], temp[9] = '0', "_", "_", "_"
            temp2 = '\t'.join(temp)
            outp2.write(temp2 + "\n")

    outp2.write('\n</code></pre>\n\n')

    # Get the English translation

    outp2.write('<pre><code class="language-conllu">\n')

    for token_num, token in enumerate(eng_sent):
        if token_num == en_token_index:  # If the english token is aligned to the main word, mark theb by *** in the english sentence too
            temp = token.conll().split('\t')
            temp[1] = "<b>***" + temp[1] + "***</b>"
            temp2 = '\t'.join(temp)
            outp2.write(temp2 + "\n")

        else:
            outp2.write(f"{token.id}\t{token.form}\t_\t_\t_\t_\t0\t_\t_\t_\n")
    outp2.write('\n</code></pre>\n\n<br>')

    # # make a table of all word alignments
    # outp2.write(
    #     f'<div> <button type="button" class="collapsible" style=\"text-align:center\">Word by word translations </button>\n<div class="content-hover">\n')
    #
    # outp2.write(
    #     f'<table><col><colgroup span="2"></colgroup>'
    #     f'<th rowspan=\"2\" style=\"text-align:center\">English</th>'
    #     f'<th rowspan=\"2\" style=\"text-align:center\">Marathi</th></tr><tr>\n')
    # for target_id, source_ids in t2s.items():
    #     target_form = tgt_sent[target_id].form
    #     target_form = target_form + ' (' + trn.transform(target_form) + ')'
    #     source_forms = []
    #     for source_id in source_ids:
    #         source_forms.append(eng_sent[source_id].form)
    #     source_form = " ".join(source_forms)
    #     outp2.write(f'<tr><td> {source_form} </td> <td> {target_form} </td></tr>\n')
    #
    # outp2.write('</table></div></div>\n<br>\n')

def readExamples():
    examples = {}
    with open(args.input + ".examples", 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            info = line.strip().split("\t")
            en_word_lemma, mr_word, sent_id, token_id, en_input_file, mr_input_file, alignment_file = info[0], info[1], info[2], info[3], info[4], info[5], info[6]
            if en_word_lemma not in examples:
                examples[en_word_lemma] = defaultdict(list)
            #mr_word = trn.transform(mr_word)
            examples[en_word_lemma][mr_word].append((int(sent_id), token_id, en_input_file, mr_input_file, alignment_file))
    return examples

def getTargetWord(en_word):
    tgt_words = src_tgt_examples.get(en_word, None)
    if not tgt_words:
        return None, -1
    sorted_tgt_words = sorted(tgt_words.items(), key=lambda kv: kv[1], reverse=True)[0]
    (tgt_word, value) = sorted_tgt_words
    return tgt_word, value


def eng_adj():
    if not os.path.exists(args.en_adj):
        with open(args.en_adj, 'w') as fout:
            for i in wn.all_synsets():
                if i.pos() in ['a', 's']:  # If synset is adj or satelite-adj.
                    synonyms = []
                    antonyms = []
                    for j in i.lemmas():  # Iterating through lemmas for each synset.
                        synonyms.append(j.name())
                        if j.antonyms():  # If adj has antonym.
                            # Prints the adj-antonym pair.
                            antonyms.append(j.antonyms()[0].name())
                        fout.write(f'name:{j.name()}, syn:{",".join(synonyms)}, ant:{",".join(antonyms)}\n')

    with open(args.en_adj, 'r') as f:
        adj = {}
        for line_num, line in f.readlines():
            if line_num % 2 !=0:
                continue
            info = line.strip().split(", ")
            name, syns, ants = info[0].split("name:")[1], info[1].split("syn:"), info[2].split("ant:")
            if len(syns) == 1:
                syns = []
            else:
                syns = syns[1].split(",")
            if len(ants) == 1:
                ants = []
            else:
                ants = ants[1].split(",")
            adj[name] = (syns, ants)

        print(f'Loading Adj: ', len(adj))
        return adj

def writeAdj():
    print(f'Writing adj')
    adjinfo = eng_adj()
    unique_words = set()
    with open(f'{helper_file_dir}/vocab_adj.html', 'w') as output:
        HEADER = ORIG_HEADER.replace("main.css", "../../main.css")
        output.write(HEADER + '\n')
        output.write(f'<ul class="nav" style="width:155%"><li class="nav"><a class="active" href=\"../../index.html\">Home</a>'
                     f'</li><li class="nav"><a href=\"../../introduction.html\">Usage</a></li>'
                     f'<li class="nav"><a href="../../about.html\">About Us</a></li></ul>\n')
        output.write(f"<br><li><a href=\"../WordUsage.html\">Back to vocabulary page</a></li>\n")
        output.write(
            f'<h3> Adjectives with synonyms and antonyms  </h3>\n')
        output.write(input_text)
        output.write(
            f'<table id="myTable" style="width:150%">'
            f'<tr><th colspan=\"rowspan=\"2\"	scope=\"colgroup\" \" style=\"text-align:center\"> English Word  </th>'
            f'<th colspan=\"rowspan=\"2\"	scope=\"colgroup\" \" style=\"text-align:center\"> Definition  </th>'
            f'<th colspan=\"rowspan=\"3\"	scope=\"colgroup\" \" style=\"text-align:center\"> Marathi Word  </th>'
             f'<th colspan=\"rowspan=\"3\"	scope=\"colgroup\" \" style=\"text-align:center\"> Synonyms  </th>'
             f'<th colspan=\"rowspan=\"3\"	scope=\"colgroup\" \" style=\"text-align:center\"> Antonyms  </th>'
            #f'<th colspan=\"rowspan=\"3\"	scope=\"colgroup\" \" style=\"text-align:center\"><b> Examples </th>'
            '</tr>\n')

        for en_word, (syns, ants) in adjinfo.items():
            en_text = f'<tr> <td> {en_word} </td>'

            synset = wn.synsets(en_word)[0]
            en_def = f'<td> {synset.definition()} </td>'

            tgt_word_text = ""

            tgt_words = src_tgt_examples.get(en_word, None)
            if not tgt_words:
                continue
            sorted_tgt_words = sorted(tgt_words.items(), key=lambda kv: kv[1], reverse=True)[0]
            (tgt_word, value) = sorted_tgt_words

            tgt_word_trn = trn.transform(tgt_word)
            tgt_word_text += f'{tgt_word} ({tgt_word_trn}), <a href="examples/{en_word}_{tgt_word_trn}.html">Examples</a>\n<br>'
            unique_words.add(f'{en_word}_{tgt_word_trn}_{tgt_word}')

            syn_text = ""
            for syn in syns:
                tgt_word, value = getTargetWord(syn)
                if value > 50 and tgt_word and len(tgt_word) > 0:
                    tgt_word_trn = trn.transform(tgt_word)
                    syn_text += f'{tgt_word} ({tgt_word_trn}), <a href="examples/{syn}_{tgt_word_trn}.html">Examples</a>\n<br>'

            ant_text = ""
            for ant in ants:
                tgt_word, value = getTargetWord(ant)
                if value > 50 and tgt_word and len(tgt_word) > 0:
                    tgt_word_trn = trn.transform(tgt_word)
                    ant_text += f'{tgt_word} ({tgt_word_trn}), <a href="examples/{ant}_{tgt_word_trn}.html">Examples</a>\n<br>'

            if len(tgt_word_text) > 0:
                output.write(f'{en_text} {en_def} <td> {tgt_word_text} </td> <td> {syn_text} </td> <td> {ant_text} </td> </tr>')
        output.write(f'</table>')

        output.write(script_text)
        output.write(FOOTER)

        output.close()


        for unique_word in unique_words:
            en_word, mr_word_trn, mr_word = unique_word.split("_")[0], unique_word.split("_")[1], unique_word.split("_")[2]
            examples_word = examples[en_word][mr_word][:10]
            example_file= f'{examples_dir}/{en_word}_{mr_word_trn}.html'
            if os.path.exists(example_file):
                continue
            with open(example_file, 'w') as fout:
                HEADER = ORIG_HEADER.replace("main.css", "../../../main.css")
                fout.write(HEADER + '\n')
                fout.write(
                    f'<ul class="nav" style="width:155%"><li class="nav"><a class="active" href=\"../../../index.html\">Home</a>'
                    f'</li><li class="nav"><a href=\"../../../introduction.html\">Usage</a></li>'
                    f'<li class="nav"><a href="../../../about.html\">About Us</a></li></ul>\n')
                fout.write(f"<br><li><a href=\"../vocab_adj.html\">Back to vocabulary</a></li>\n")
                fout.write(
                    f'<h3> Example usages of <b> {mr_word} ({mr_word_trn}) </b> </h3>\n')
                prev_file = ""
                for (sent_id, token_id, en_input_file, mr_input_file, alignment_file) in examples_word:
                    if prev_file != en_input_file:
                        en_data = pyconll.load_from_file(f'{en_input_file}')
                        mr_data = pyconll.load_from_file(f'{mr_input_file}')
                        with open(alignment_file, 'r') as fin:
                            alignmentdata = fin.readlines()
                    en_sent, mr_sent = en_data[sent_id], mr_data[sent_id]
                    alignmentsent = alignmentdata[sent_id]

                    outputExamplesWebPage(fout, en_sent, mr_sent, token_id, alignmentsent)

                fout.write(f'</div>')
                fout.write(FOOTER)

def readVocabFile():
    if os.path.exists(args.input) and os.path.exists(args.input + ".examples"):
        print(f'Existing ', args.output)
        return
    word_pairs = {}
    examples = {}
    en_input_file, mr_input_file, alignment_input_file = args.en_input, args.mr_input, args.alignment
    with open(alignment_input_file, 'r') as fin:
        alignmentdata = fin.readlines()

    if not os.path.exists(en_input_file) or not os.path.exists(mr_input_file):
        print(f'Input files: {en_input_file} or {mr_input_file} not present!')
        exit(-1)

    print('Processing: ', en_input_file, mr_input_file)
    data = pyconll.load_from_file(en_input_file)
    mr_data = pyconll.load_from_file(mr_input_file)

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
            if len(examples[en_word_lemma][mr_words_text]) < 10:
                examples[en_word_lemma][mr_words_text].append(
                    (sent_num, token.id, en_input_file, mr_input_file, alignment_input_file))


    ''' 
    Code to read multiple files if a single file is large, change the below code depending on your file naming format.
    Below code works when train file is named as for English sentences (train.en.aa.conllu), its corresponding Marathi sentences (train.mr.aa.conllu), and its alignment (train.en.mr.pred.aa)
    for [path, dir, files] in os.walk(args.en_input):
        for file in files:
            if "train.en." not in file:
                continue
            prefix = file.split("train.en.")[1].split(".conllu")[0]
            en_input_file, mr_input_file, alignment_input_file = \
                f'{args.en_input}/train.en.{prefix}.conllu', \
                f'{args.mr_input}/train.kn.sud.{prefix}.conllu', \
                f'{args.alignment}/train.en.kn.pred.{prefix}'

            with open(alignment_input_file, 'r') as fin:
                alignmentdata = fin.readlines()

            if not os.path.exists(en_input_file) or not os.path.exists(mr_input_file):
                continue

            print('Processing: ', en_input_file, mr_input_file)
            data = pyconll.load_from_file(en_input_file)
            mr_data = pyconll.load_from_file(mr_input_file)

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
                    if len(examples[en_word_lemma][mr_words_text]) < 10:
                        examples[en_word_lemma][mr_words_text].append(
                            (sent_num, token.id, en_input_file, mr_input_file, alignment_input_file))
    '''

    with open(args.input, 'w') as fout, open(args.input + ".examples", 'w') as fexample:
        for en_word_lemma, mr_words in word_pairs.items():
            mr_words_string = []

            for mr_word, value in mr_words.items():
                mr_words_string.append(f'{mr_word}={value}')
                if value > 50:
                    for (sent, id, en_input_file, mr_input_file, alignment_input_file) in examples[en_word_lemma][
                                                                                              mr_word][:10]:
                        fexample.write(
                            f'{en_word_lemma}\t{mr_word}\t{sent}\t{id}\t{en_input_file}\t{mr_input_file}\t{alignment_input_file}\n')

            fout.write(f'en:{en_word_lemma}\t{";".join(mr_words_string)}\n')


if __name__ == '__main__':
    language_fullname = args.lang
    lang_full = args.lang
    lang_id = args.lang
    trn = Transliterator(source=args.transliterate, target='eng', build_lookup=True)

    readVocabFile()

    with open(args.input, 'r') as fin:
        lines = fin.readlines()
        src_tgt_examples = {}
        for line in lines:
            if line.startswith("en"):
                info = line.strip().split("\t")
                src_word = info[0].split("en:")[1]
                tgt_words = info[1].split(";")
                if src_word not in src_tgt_examples:
                    src_tgt_examples[src_word] = {}
                for tgt_word_info in tgt_words:
                    if len(tgt_word_info.split("=")) != 2:
                        continue
                    tgt_word = tgt_word_info.split("=")[0]
                    value = int(tgt_word_info.split("=")[1])
                    src_tgt_examples[src_word][tgt_word] = value


    examples = readExamples()


    helper_file_dir = f'{args.folder_name}/{args.lang}/WordUsage/'
    examples_dir = f'{helper_file_dir}/examples/'
    os.system(f'mkdir -p {examples_dir}')
    writeAdj()

