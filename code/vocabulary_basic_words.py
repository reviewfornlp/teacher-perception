from nltk.corpus import wordnet as wn
import argparse
from collections import defaultdict
from indictrans import Transliterator
import os
import pyconll
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="vocab_wordnet.txt")
parser.add_argument("--folder_name", type=str, default='./website/')
parser.add_argument("--lang", type=str, default="kn_en")
parser.add_argument("--transliterate", type=str, default="kan")
parser.add_argument("--en_adj", type=str, default="en_adj.txt")
args = parser.parse_args()

required_synsets = ['food.n.01', 'food.n.02', 'relative.n.01', 'animal.n.01', 'fruit.n.01', 'color.n.01', 'day.n.04', 'sit.v.01', 'talk.v.01', 'run.v.01', 'eat.v.01'
                    ,'ask.v.02', 'body_part.n.01', 'vehicle.n.01', 'water.n.01', 'ocean.n.01', 'fire.n.01', 'air.n.01', 'cook.v.01', 'flower.n.01', 'professional.n.01', 'furniture.n.01', 'clothing.n.01']
remove = ['entity.n.01', 'abstraction.n.06', 'psychological_feature.n.01', 'physical_entity.n.01', 'causal_agent.n.01', 'person.n.01', 'object.n.01', 'whole.n.02', 'artifact.n.01', 'event.n.01', 'act.n.02', 'attribute.n.02', 'state.n.02']
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

def outputWebpage(folder_name, lang):
    with open(f'{folder_name}/vocab.html', 'w') as output:
        HEADER = ORIG_HEADER.replace("main.css", "../../main.css")
        output.write(HEADER + '\n')
        output.write(f'<ul class="nav" style="width:155%"><li class="nav"><a class="active" href=\"../../index.html\">Home</a>'
                     f'</li><li class="nav"><a href=\"../../introduction.html\">Usage</a></li>'
                     f'<li class="nav"><a href="../../about.html\">About Us</a></li></ul>\n')
        output.write(f"<br><li><a href=\"../../index.html\">Back to {lang} page</a></li>\n")
        output.write(
            f'<h3> Vocabulary covering <b>nouns (n), verbs (v), adjectives (a), adverbs (r)</b>  </h3>\n')
        output.write(input_text)
        output.write(
            f'<table id="myTable" style="width:150%">'
            f'<tr><th colspan=\"rowspan=\"2\"	scope=\"colgroup\" \" style=\"text-align:center\"> Type  </th>'
            f'<th colspan=\"rowspan=\"2\"	scope=\"colgroup\" \" style=\"text-align:center\"> Definition  </th>'
            f'<th colspan=\"rowspan=\"3\"	scope=\"colgroup\" \" style=\"text-align:center\"> Words  </th>'
            #f'<th colspan=\"rowspan=\"3\"	scope=\"colgroup\" \" style=\"text-align:center\"><b> Examples </th>'
            '</tr>\n')

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

def eng_adj():
    with open(args.en_adj, 'r') as f:
        adj = {}
        for line in f.readlines():
            info = line.strip().split(", ")
            name, syns, ants = info[0].split("name:")[1], info[1].split("syn:")[1], info[2].split("ant:")[1]
            adj[name] = (syns, ants)
        print(f'Loading Adj: ', len(adj))
        return adj

def writeAdj():
    print(f'Writing adj')
    adjinfo = eng_adj()
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
            f'<tr><th colspan=\"rowspan=\"2\"	scope=\"colgroup\" \" style=\"text-align:center\"> Type  </th>'
            f'<th colspan=\"rowspan=\"2\"	scope=\"colgroup\" \" style=\"text-align:center\"> Definition  </th>'
            f'<th colspan=\"rowspan=\"3\"	scope=\"colgroup\" \" style=\"text-align:center\"> Words  </th>'
            #f'<th colspan=\"rowspan=\"3\"	scope=\"colgroup\" \" style=\"text-align:center\"><b> Examples </th>'
            '</tr>\n')

        for en_word, (syns, ants) in adjinfo.items():
            en_text = f'<tr> <td> {en_word} </td>'

            synset = wn.synsets(en_word)[0]
            en_def = f'<td> {synset.definition()} </td>'

            tgt_word_text = ""
            tgt_words_covered = set()

            tgt_words = src_tgt_examples.get(en_word, None)
            if not tgt_words:
                continue
            sorted_tgt_words = sorted(tgt_words.items(), key=lambda kv: kv[1], reverse=True)[0]
            (tgt_word, value) = sorted_tgt_words

            if value > 50 and tgt_word and len(tgt_word) > 0:
                en_tgt_word = f'{en_word}-{tgt_word}'
                if en_tgt_word not in tgt_words_covered:
                    tgt_word_trn = trn.transform(tgt_word)
                    tgt_word_text += f'{en_word} -- {tgt_word} ({tgt_word_trn}), <a href="examples/{en_word}_{tgt_word_trn}.html">Examples</a>\n<br>'
                    unique_words.add(f'{en_word}_{tgt_word_trn}_{tgt_word}')
                tgt_words_covered.add(en_tgt_word)
            if len(tgt_word_text) > 0:
                output.write(f'{en_text} {en_def} <td> {tgt_word_text} </td> </tr>')
        output.write(f'</table>')

        output.write(script_text)
        output.write(FOOTER)

        output.close()


        for unique_word in unique_words:
            en_word, mr_word_trn, mr_word = unique_word.split("_")[0], unique_word.split("_")[1], unique_word.split("_")[2]
            examples_word = examples[en_word][mr_word][:10]
            with open(f'{examples_dir}/{en_word}_{mr_word_trn}.html', 'w') as fout:
                HEADER = ORIG_HEADER.replace("main.css", "../../../main.css")
                fout.write(HEADER + '\n')
                fout.write(
                    f'<ul class="nav" style="width:155%"><li class="nav"><a class="active" href=\"../../../index.html\">Home</a>'
                    f'</li><li class="nav"><a href=\"../../../introduction.html\">Usage</a></li>'
                    f'<li class="nav"><a href="../../../about.html\">About Us</a></li></ul>\n')
                fout.write(f"<br><li><a href=\"../../vocab.html\">Back to vocabulary</a></li>\n")
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
if __name__ == '__main__':
    language_fullname = args.lang
    lang_full = args.lang
    lang_id = args.lang
    trn = Transliterator(source=args.transliterate, target='eng', build_lookup=True)

    helper_file_dir = f'{args.folder_name}/{args.lang}/WordUsage/'
    examples_dir = f'{helper_file_dir}/examples/'
    os.system(f'mkdir -p {examples_dir}')
    output = outputWebpage(helper_file_dir, args.lang)

    with open(args.input, 'r') as fin:
        lines = fin.readlines()
        wsd_examples = defaultdict(list)
        src_tgt_examples = {}
        wsd_depths = {}
        for line in lines:
            if line.startswith("wsd"):
                info = line.strip().split("\t")
                wsd_info = info[0].split("wsd:")[1]
                wsd_depth = info[1].split("depth:")[1]
                en_examples = info[2].split(",")
                wsd_examples[wsd_info] = en_examples
                wsd_depths[wsd_info] = wsd_depth
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

    all_synsets = []
    for synset in wsd_examples.keys():
        depth = int(wsd_depths[synset])
        if depth < 4 or depth >= 5:
            continue
        all_synsets.append(synset)

    other_synsets = list(set(all_synsets) - set(required_synsets) - set(remove))
    other_synsets.sort()
    #required_synsets = required_synsets + other_synsets
    print(len(required_synsets))


    hypo = lambda s: s.hyponyms()
    unique_words = set()
    with open(f'{helper_file_dir}/vocab.html', 'a') as output:
        for synset_name in required_synsets:

            category_name = synset_name.split(".")[0]
            type = synset_name.split(".")[1]
            en_text = f'<tr> <td> {category_name} ({type}) </td>'

            synset = wn.synset(synset_name)
            en_def = f'<td> {synset.definition()} </td>'
            synset_types = [synset] + list(synset.closure(hypo))
            tgt_word_text = ""
            tgt_words_covered = set()
            for synset_type in synset_types:
                synset_type_name = synset_type._name
                if synset_type_name in wsd_examples:

                    en_examples = wsd_examples[synset_type_name]

                    for en_word in en_examples:
                        tgt_words = src_tgt_examples[en_word]
                        sorted_tgt_words = sorted(tgt_words.items(), key=lambda kv:kv[1], reverse=True)[0]
                        (tgt_word, value) = sorted_tgt_words
                        if value > 50 and tgt_word and len(tgt_word) > 0:
                            en_tgt_word = f'{en_word}-{tgt_word}'
                            if en_tgt_word not in tgt_words_covered:
                                tgt_word_trn = trn.transform(tgt_word)
                                tgt_word_text += f'{en_word} -- {tgt_word} ({tgt_word_trn}), <a href="examples/{en_word}_{tgt_word_trn}.html">Examples</a>\n<br>'
                                unique_words.add(f'{en_word}_{tgt_word_trn}_{tgt_word}')
                            tgt_words_covered.add(en_tgt_word)
            if len(tgt_word_text) > 0:
                output.write(f'{en_text} {en_def} <td> {tgt_word_text} </td> </tr>')
        output.write(f'</table>')

        output.write(script_text)
        output.write(FOOTER)

        output.close()


        for unique_word in unique_words:
            en_word, mr_word_trn, mr_word = unique_word.split("_")[0], unique_word.split("_")[1], unique_word.split("_")[2]
            examples_word = examples[en_word][mr_word][:10]
            with open(f'{examples_dir}/{en_word}_{mr_word_trn}.html', 'w') as fout:
                HEADER = ORIG_HEADER.replace("main.css", "../../../main.css")
                fout.write(HEADER + '\n')
                fout.write(
                    f'<ul class="nav" style="width:155%"><li class="nav"><a class="active" href=\"../../../index.html\">Home</a>'
                    f'</li><li class="nav"><a href=\"../../../introduction.html\">Usage</a></li>'
                    f'<li class="nav"><a href="../../../about.html\">About Us</a></li></ul>\n')
                fout.write(f"<br><li><a href=\"../../vocab.html\">Back to vocabulary</a></li>\n")
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

    writeAdj()



