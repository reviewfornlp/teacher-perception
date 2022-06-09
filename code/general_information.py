import argparse, pyconll, utils, os
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from collections import defaultdict
from indictrans import Transliterator

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default='data//')
parser.add_argument("--relation_map", type=str, default="./relation_map")
parser.add_argument("--folder_name", type=str, default='./syntax_lex/',
                    help="Folder to hold the rules, need to add header.html, footer.html always")
parser.add_argument("--lang", type=str, default="kn_en")
parser.add_argument("--distributed_files", type=str, default=None,
                    help="Essentially split the large training data in multiple files, specify the path to one "
                         "such file, it will load all files separately instead of loading one large file in memoty")
parser.add_argument("--transliterate", type=str, default=None)
parser.add_argument("--sud", action="store_true", default=True, help="Enable to read from SUD treebanks")
parser.add_argument("--auto", action="store_true", default=False, help="Enable to read from automatically parsed data")
parser.add_argument("--noise", action="store_true", default=False,
                    help="Enable to read from automatically parsed data")

args = parser.parse_args()

lang_map = {'SUD_Italian-VIT': 'it_vit', 'SUD_Turkish-IMST': 'tr_imst', 'SUD_Russian-GSD': 'ru_gsd',
            'SUD_Norwegian-Nynorsk': 'no_nynorsk', 'SUD_French-GSD': 'fr_gsd', 'SUD_Uyghur-UDT': 'ug_udt',
            'SUD_Romanian-Nonstandard': 'ro_nonstandard', 'SUD_Bulgarian-BTB': 'bg_btb', 'SUD_Galician-CTG': 'gl_ctg',
            'SUD_Czech-PUD': 'cs_pud', 'SUD_Finnish-TDT': 'fi_tdt', 'SUD_Polish-PDB': 'pl_lfg',
            'SUD_Latin-ITTB': 'la_ittb', 'SUD_Slovenian-SSTJ': 'sl_sst', 'SUD_Chinese-GSD': 'zh_cfl',
            'SUD_Skolt_Sami-Giellagas': 'sms_giellagas', 'SUD_Dutch-Alpino': 'nl_alpino',
             'SUD_Catalan-AnCora': 'ca_ancora', 'SUD_Bhojpuri-BHTB': 'bho_bhtb', 'SUD_Urdu-UDTB': 'ur_udtb',
            'SUD_Amharic-ATT': 'am_att', 'SUD_Hindi-HDTB': 'hi_hdtb', 'SUD_Naija-NSC': 'pcm_nsc', 'SUD_Komi_Zyrian-IKDP': 'kpv_ikdp',
            'SUD_Indonesian-GSD': 'id_pud', 'SUD_Serbian-SET': 'sr_set', 'SUD_Basque-BDT': 'eu_bdt',
            'SUD_Lithuanian-ALKSNIS': 'lt_hse', 'SUD_Vietnamese-VTB': 'vi_vtb',
            'SUD_Tagalog-TRG': 'tl_trg', 'SUD_Persian-Seraji': 'fa_seraji',
            'SUD_North_Sami-Giella': 'sme_giella',
            'SUD_Danish-DDT': 'da_ddt', 'SUD_Swedish_Sign_Language-SSLC': 'swl_sslc',
            'SUD_Arabic-NYUAD': 'ar_nyuad',
            'SUD_Scottish_Gaelic-ARCOSG': 'gd_arcosg', 'SUD_Ancient_Greek-PROIEL': 'grc_proiel', 'SUD_German-GSD': 'de_gsd',
            'SUD_Moksha-JR': 'mdf_jr', 'SUD_Telugu-MTG': 'te_mtg', 'SUD_Maltese-MUDT': 'mt_mudt', 'SUD_Wolof-WTB': 'wo_wtb',
            'SUD_Japanese-PUD': 'ja_pud', 'SUD_Assyrian-AS': 'aii_as', 'SUD_Hebrew-HTB': 'he_htb', 'SUD_Portuguese-GSD': 'pt_gsd',
            'SUD_Old_Church_Slavonic-PROIEL': 'cu_proiel', 'SUD_Livvi-KKPP': 'olo_kkpp',
            'SUD_Erzya-JR': 'myv_jr',
            'SUD_Welsh-CCG': 'cy_ccg',
            'SUD_Belarusian-HSE': 'be_hse',
            'SUD_Korean-PUD': 'ko_pud', 'SUD_Swedish-LinES': 'sv_lines',
            'SUD_Ukrainian-IU': 'uk_iu',
            'SUD_Buryat-BDT': 'bxr_bdt',
            'SUD_Tamil-TTB': 'ta_ttb',
            'SUD_Irish-IDT': 'ga_idt', 'SUD_Slovak-SNK': 'sk_snk', 'SUD_Hungarian-Szeged': 'hu_szeged',
            'SUD_Gothic-PROIEL': 'got_proiel', 'SUD_Croatian-SET': 'hr_set',
            'SUD_Akkadian-PISANDUB': 'akk_pisandub',
            'SUD_Greek-GDT': 'el_gdt', 'SUD_Classical_Chinese-Kyoto': 'lzh_kyoto', 'SUD_Coptic-Scriptorium': 'cop_scriptorium',
            'SUD_Latvian-LVTB': 'lv_lvtb', 'SUD_Warlpiri-UFAL': 'wbp_ufal', 'SUD_Sanskrit-UFAL': 'sa_ufal',
            'SUD_Mbya_Guarani-Thomas': 'gun_thomas', 'SUD_Kazakh-KTB': 'kk_ktb', 'SUD_Estonian-EDT': 'et_edt',
            'SUD_Old_French-SRCMF': 'fro_srcmf', 'SUD_Upper_Sorbian-UFAL': 'hsb_ufal', 'SUD_Bambara-CRB': 'bm_crb',
            'SUD_Afrikaans-AfriBooms': 'af_afribooms', 'SUD_Cantonese-HK': 'yue_hk', 'SUD_Armenian-ArmTDP': 'hy_armtdp'}


with open(f"{args.folder_name}/header.html") as inp:
    ORIG_HEADER = inp.readlines()
ORIG_HEADER = ''.join(ORIG_HEADER)

with open(f"{args.folder_name}/footer.html") as inp:
    FOOTER = inp.readlines()
FOOTER = ''.join(FOOTER)

def printPOSMorph(fileoutp):
    fileoutp.write(f'<h2> The popular grammar categories observed in the corpus. Click on each to explore some example words. </h2>')
    fileoutp.write(f'<table>'
               f'<tr><th rowspan=\"2\" style=\"text-align:center\"> Grammar Category </th>'
                   f'<th rowspan=\"2\" style=\"text-align:center\"> Distribution of POS within each category </th>'
               f'<th rowspan=\"2\" style=\"text-align:center\"> Example words (per POS) for each grammar category </th><tr></tr>'
               f'</tr>')
    # sort the pos by count
    for feature in required_features:
        if feature not in pos_count:
            continue
        sorted_pos = sorted(pos_count[feature].items(), key=lambda kv: kv[1], reverse=True)
        pos_to_id, pos_order = {}, []
        for (pos, _) in sorted_pos:
            pos_to_id[pos] = len(pos_to_id)
            pos_order.append(pos)

        # Stacked histogram
        feature_features_set, feature_pos_barplots = features_set[feature], pos_barplots[feature]
        sns.set()
        bars_num = np.zeros((len(feature_features_set), len(feature_pos_barplots)))
        x_axis = []
        feat_to_id, id_to_feat = utils.get_vocab_from_set(feature_features_set)

        for pos in pos_order:
            feats = feature_pos_barplots[pos]
            x_axis.append(pos)
            pos_id = pos_to_id[pos]
            for feat, num in feats.items():
                feat_id = feat_to_id[feat]
                bars_num[feat_id][pos_id] = num

        r = [i for i in range(len(pos_to_id))]
        handles, color = [], ['steelblue', 'orange', 'olivedrab', 'peru', 'seagreen', 'chocolate', 'tan',
                              'lightseagreen', 'green', 'teal', 'tomato', 'lightgreen', 'yellow', 'lightblue', 'azure',
                              'red', 'aqua']
        bars = np.zeros((len(feature_pos_barplots)))
        for barnum in range(len(feature_features_set)):
            plt.bar(r, bars_num[barnum], bottom=bars, color=color[barnum], edgecolor='white', width=1)
            handles.append(mpatches.Patch(color=color[barnum], label=id_to_feat[barnum]))
            bars += bars_num[barnum]

        handles.reverse()
        # Custom X axis
        plt.xticks(r, x_axis, rotation=45, fontsize=9)
        plt.xlabel("pos")
        plt.ylabel("count")
        plt.legend(handles=handles)
        # plt.xticks(rotation=45)
        plt.savefig(f"{helper_file_dir}/{feature}_pos.png")
        plt.close()

        fileoutp.write(f'<tr> <td> {feature} </td> <td><img src=\"{feature}_pos.png\" alt=\"{feature}\"  height=400 width=500></td>')

        feature_forms_num, feature_lemma_freq, feature_lemma_inflection = forms_num[feature], lemma_freq[feature], \
                                                                          lemma_inflection[feature]

        fileoutp.write(f'<td>')
        for pos, pos_dict in feature_forms_num.items():
            pos_string, link = get_relation(pos)
            filename = f"{helper_file_dir}/{feature}_{pos}.html"
            with open(filename, 'w') as outp:
                HEADER = ORIG_HEADER.replace("main.css", "../..//main.css")
                outp.write(HEADER + '\n')
                outp.write(f'<ul class="nav" style="width:155%"><li class="nav"><a class="active" href=\"../..//index.html\">Home</a>'
                               f'</li><li class="nav"><a href=\"../..//introduction.html\">Usage</a></li>'
                               f'<li class="nav"><a href="../..//about.html\">About Us</a></li></ul>\n')
                outp.write(f"<br><li><a href=\"./syntactic_info.html\">Back to syntactic page</a></li>\n")
                outp.write(f"<h1> Examples of <b>{pos_string}</b> words for  each {feature} value : </h1>")
                outp.write(f"<h1> For detailed definition of what a </b>{pos_string}</b> means, check <a href=\"{link}\"> here </a>. </h1>")
                outp.write(
                    f"<p> The word types shown below are sorted by token frequency and further grouped by lemma.</p>")
                # Sort the tokens within a pos for a feature
                feature_values = pos_dict.keys()
                input_text = f'<input type="text" id="myInput" class="myInput" style="width=155%" onkeyup="myFunction()" placeholder="Search for a word (e.g. to for तो or त)" title="word">\n'
                script_text = '\n<script> \n function myFunction() {  var input, filter, table, tr, td, i, txtValue;\n' \
                              'input = document.getElementById("myInput");\n' \
                              'filter = input.value.toLowerCase();\n' \
                              'table = document.getElementById("myTable");\n' \
                              'tr = table.getElementsByTagName("tr");\n' \
                              'for (i = 0; i < tr.length; i++) {\n' \
                              'var colcount = tr[i].getElementsByTagName("td").length;\n' \
                              'td = tr[i].getElementsByTagName("td")[0];\n' \
                              'if (td) {\n' \
                              'txtValue = td.textContent || td.innerText;\n' \
                              'if (txtValue.toLowerCase().indexOf(filter) > -1) {\n' \
                              'tr[i].style.display = "";\n' \
                              '} else { \n' \
                              'tr[i].style.display = "none"; \n }}}} \n ' \
                              '</script>'
                outp.write(f'{input_text}')
                outp.write(f'<table id="myTable" style="width:155%"><col><colgroup span=\"{len(feature_values) + 4} \"></colgroup>'
                           f'<tr><th rowspan=\"2\" style=\"text-align:center\">Lemma</th>'
                           f'<th rowspan=\"2\" style=\"text-align:center\"> Morphosyntactic <br> Attributes</th>'
                           f'<th colspan=\"{len(feature_values)}	scope=\"colgroup\" \" style=\"text-align:center\">{feature}</th>'
                           f'<th colspan="2"	scope=\"colgroup\" style=\"text-align:center\">  </th></tr><tr>')

                #for p in agree:
                for feat in feature_values:
                    outp.write(f'<th scope=\"col\"> {feat} </th>')
                outp.write('</tr>')
                if len(feature_values) > 1:
                    fileoutp.write(f'<a href="{feature}_{pos}.html"> {pos}, <br>')
                # Sort the tokens within a pos using lemma
                sorted_lemma_dict = sorted(feature_lemma_freq[pos].items(), key=lambda kv: kv[1], reverse=True)[
                                    :30]
                for (lemma, _) in sorted_lemma_dict:
                    if trn:
                        lemma_ = trn.transform(lemma)
                        lemma_string = f'{lemma} ({lemma_})'
                    else:
                        lemma_ = lemma
                        lemma_string = f'{lemma}'

                    for inflection in feature_lemma_inflection[pos][lemma].keys():
                        outp.write(f'<tr><td> {lemma_string} </td>\n')
                        outp.write(f'<td> {inflection} </td>')
                        inflection_feature_value = feature_lemma_inflection[pos][lemma][inflection]
                        for feat in feature_values:
                            if feat in inflection_feature_value:
                                if trn:
                                    inflection_value_string = f'{inflection_feature_value[feat]} ({trn.transform(inflection_feature_value[feat])})'
                                else:
                                    inflection_value_string = f'{inflection_feature_value[feat]}'

                                outp.write(f'<td> {inflection_value_string} </td>')
                            else:
                                outp.write(f'<td> - </td> ')
                        #print(lemma, lemma_, )
                        outp.write(f'<td> <a href="examples/examples_{lemma_}.html"> Examples </a> </td>\n')
                        writeExamples(examples_per_lemma, lemma, lemma_)
                        outp.write('</tr>\n')
                outp.write('</table>')
                outp.write(f'{script_text}\n')
                outp.write(FOOTER)
                print(filename)
                #exit(-1)

        fileoutp.write(f'</td>')

    fileoutp.write("</table>" + "\n")

def writeExamples(examples_per_lemma, lemma, lemma_):
    examples = examples_per_lemma[lemma_][:20]
    filename = f'{helper_file_dir}/examples/examples_{lemma_}.html'
    if os.path.exists(filename):
        return
    try:
        with open(filename, 'w') as outp2:
            HEADER = ORIG_HEADER.replace("main.css", "../../../main.css")
            outp2.write(HEADER + '\n')
            outp2.write(
                f'<ul class="nav" style="width:155%"><li class="nav"><a class="active" href=\"../../../index.html\">Home</a>'
                f'</li><li class="nav"><a href=\"../../../introduction.html\">Usage</a></li>'
                f'<li class="nav"><a href="../../../about.html\">About Us</a></li></ul>\n')
            outp2.write(f"<br><li><a href=\"../syntactic_info.html\">Back to syntactic page</a></li>\n")
            outp2.write(f"<h1> Examples of the root word <b>{lemma} ({lemma_})</b> also marked by ***, hover on that word to see more grammar properties. </h1>\n")
            for (sent, tokid) in examples:
                outp2.write('<pre><code class="language-conllu">\n')
                for token in sent:
                    if trn:
                        transliterated = trn.transform(token.form)
                    else:
                        transliterated = None
                    if token.id == tokid:
                        temp = token.conll().split('\t')
                        if transliterated:
                            temp[1] = "***" + temp[1] + '(' + transliterated + ')' + "***"
                        else:
                            temp[1] = "***" + temp[1] + "***"
                        temp2 = '\t'.join(temp)
                        outp2.write(temp2 + "\n")
                    elif '-' not in token.id:
                        temp = token.conll().split('\t')
                        if transliterated:
                            temp[1] = temp[1] + '(' + transliterated + ')'
                        else:
                            temp[1] = temp[1]

                        temp[6], temp[7], temp[8], temp[9] = '0', "_", "_", "_"
                        temp2 = '\t'.join(temp)
                        outp2.write(temp2 + "\n")
                outp2.write('\n</code></pre>\n\n')
            outp2.write(f'</div>')
            outp2.write(FOOTER)
    except:
        return

def get_relation(info):
    if info in relation_map:
        value, link = relation_map.get(info, info)
    elif info.lower() in relation_map:
        value, link = relation_map.get(info.lower())
    elif info.split("@")[0] in relation_map:
        value, link = relation_map.get(info.split("@")[0], info)
    elif info.split("@")[0].lower() in relation_map:
        value, link = relation_map.get(info.split("@")[0].lower(), info)
    else:
        value, link = info, "https://surfacesyntacticud.github.io/guidelines/u/"
    return value, link

if __name__ == "__main__":
    treebank_map = {v:k for k,v in lang_map.items()}

    # if args.lang not in treebank_map: #To ensure that the language belongs to the SUD treebanks
    #     exit(-1)
    #args.input = args.input + f'{treebank_map[args.lang]}'
    relation_map = {}
    with open(args.relation_map, "r") as inp:
        for line in inp.readlines():
            info = line.strip().split(";")
            key = info[0]
            value = info[1]
            if len(info) == 3:
                link = info[2]
            else:
                link = "https://universaldependencies.org/"
            relation_map[key] = (value, link)
            relation_map[key.lower()] = (value, link)
            if '@x' in key:
                relation_map[key.split("@x")[0]] = (value, link)


    if args.distributed_files:
        filenames = utils.getSmallerFiles(args.distributed_files)
        filenames = filenames
        print(filenames)
        #exit(-1)
    else:
        train_path, _, _, lang = utils.getTreebankPaths(args.input.strip(), args)
        if train_path is None:
            print(f'Skipping the treebank as no training data!')
            exit(-1)
        filenames = [train_path]

    language_fullname = args.lang
    lang_full = args.lang
    lang_id = args.lang
    if args.transliterate:
        trn = Transliterator(source=args.transliterate, target='eng', build_lookup=True)
    else:
        trn = None

    helper_file_dir = f'{args.folder_name}/{lang_full}/helper/'
    try:
        os.system(f'mkdir -p {helper_file_dir}')
    except:
        i = 0
    try:
        os.system(f'mkdir -p {helper_file_dir}/examples/')
    except:
        i = 0

    #Accumulate examples for each POS tag and deprel
    tokens, lemmas, lemmaGroups,   lemma_freq, lemma_inflection, pos_barplots, features_set, \
    pos_count, pos_values, feature_values, forms_num = [], {}, defaultdict(set), {}, {}, {}, defaultdict(set), \
                                            {}, [], defaultdict(list), {}
    required_features = ['Gender', 'Person', 'Number', 'Case', 'Tense']
    deprel = defaultdict(lambda : 0)
    deprel_examples = {}
    deprel_words_examples = {}
    pos_examples = {}

    for filename in filenames:
        #try:
        f = filename.strip()
        train_data = pyconll.load_from_file(f"{f}")
        print(f)
        examples_per_lemma = defaultdict(list)
        for sent_num, sentence in enumerate(train_data):
            for token_num, token in enumerate(sentence):
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
                if not lemma or not token.form:
                    continue


                lemmas[token.form.lower()] = lemma
                lemmaGroups[lemma].add(token.form.lower())
                if lemma:
                    if trn:
                        lemma_ = trn.transform(lemma)
                    else:
                        lemma_ = lemma
                    examples_per_lemma[lemma_].append((sentence, token.id))

                for feature in required_features:
                    if feature not in pos_barplots:
                        pos_barplots[feature] = {}
                        pos_count[feature] = defaultdict(lambda : 0)
                        forms_num[feature] = {}
                        lemma_freq[feature] = {}
                        lemma_inflection[feature] = {}
                    if pos not in pos_barplots[feature]:
                        pos_barplots[feature][pos] = defaultdict(lambda: 0)
                        forms_num[feature][pos] = {}
                        lemma_freq[feature][pos] = defaultdict(lambda: 0)
                        lemma_inflection[feature][pos] = {}
                        pos_examples[pos] = defaultdict(lambda : 0)

                    if lemma:
                        lemma_freq[feature][pos][lemma.lower()] += 1
                    if lemma and lemma.lower() not in lemma_inflection[feature][pos]:
                        lemma_inflection[feature][pos][lemma.lower()] = {}
                    pos_count[feature][pos] += 1
                    pos_examples[pos][token.form.lower()] += 1

                    # Aggregae morphology properties of required-properties - feature
                    morphology_props = set(required_features) - set([feature])
                    morphology_prop_values = []
                    for morphology_prop in morphology_props:
                        if morphology_prop in feats:
                            morphology_prop_values.append(",".join(feats[morphology_prop]))
                    morphology_prop_values.sort()
                    inflection = ";".join(morphology_prop_values)
                    if lemma and inflection not in lemma_inflection[feature][pos][lemma.lower()]:
                        lemma_inflection[feature][pos][lemma.lower()][inflection] = {}

                    if feature in feats:
                        values = list(feats[feature])
                        values.sort()
                        feature_values[feature].append(",".join(values))

                    else:
                        values = ['NA']
                        feature_values[feature].append("NA")

                    for feat in values:
                        features_set[feature].add(feat)
                        pos_barplots[feature][pos][feat] += 1
                        if feat not in forms_num[feature][pos]:
                            forms_num[feature][pos][feat] = defaultdict(lambda : 0)
                        forms_num[feature][pos][feat][token.form.lower()] += 1
                        if lemma:
                            lemma_inflection[feature][pos][lemma.lower()][inflection][feat] = token.form.lower()

                if relation:
                    deprel[relation] += 1
                    if relation not in deprel_words_examples:
                        deprel_words_examples[relation] = defaultdict(lambda : 0)
                        deprel_examples[relation] = defaultdict(list)
                    head = token.head
                    if head and head != '0':
                        if trn:
                            dep_head = f'{token.form.lower()} ({trn.transform(token.form)}) -- {sentence[token.head].form.lower()} ({trn.transform(sentence[token.head].form)})'
                        else:
                            dep_head = f'{token.form.lower()} -- {sentence[token.head].form.lower()}'

                        deprel_words_examples[relation][dep_head] += 1
                        if len(sentence) < 8:
                            deprel_examples[relation][dep_head].append((filename, sent_num, token_num))
        # except:
        #     print(f'Error in {filename}',)

    filename = f"{helper_file_dir}/syntactic_info.html"
    with open(filename, 'w') as fileoutp:
        HEADER = ORIG_HEADER.replace("main.css", "../../main.css")
        fileoutp.write(HEADER + '\n')
        fileoutp.write(f'<ul class="nav"><li class="nav"><a class="active" href=\"../../index.html\">Home</a>'
                     f'</li><li class="nav"><a href=\"../..//introduction.html\">Usage</a></li>'
                     f'<li class="nav"><a href="../..//about.html\">About Us</a></li></ul>\n')
        fileoutp.write(f"<br><li><a href=\"../..//index.html\">Back to home page</a></li>\n")
        fileoutp.write(f'<h1> You can explore the following different syntactic properties of the languages below. </h1>')
        fileoutp.write(
            f'<h2> The different grammar relations can be found <a href="./relations.html"> here </a> </h2>')

        printPOSMorph(fileoutp)


        # create a page for each relatio
        sorted_relations = sorted(deprel.items(), key=lambda kv: kv[1], reverse=True)

        filename = f"{helper_file_dir}/relations.html"
        with open(filename, 'w') as outp:
            HEADER = ORIG_HEADER.replace("main.css", "../..//main.css")
            outp.write(HEADER + '\n')
            outp.write(f'<ul class="nav"><li class="nav"><a class="active" href=\"../..//index.html\">Home</a>'
                           f'</li><li class="nav"><a href=\"../..//introduction.html\">Usage</a></li>'
                           f'<li class="nav"><a href="../..//about.html\">About Us</a></li></ul>\n')
            outp.write(f"<br><li><a href=\"./syntactic_info.html\">Back to syntactic information page</a></li>\n")

            outp.write(f"<h1> Examples of different relations observed in {language_fullname} </h1>")
            outp.write(
                f"<p> Click on each relation to check out examples constructions!")
            outp.write(f'<table><col><colgroup span=\"{len(feature_values)}\"></colgroup>'
                       f'<tr><th rowspan=\"2\" style=\"text-align:center\">Relation</th>'
                       f'<th rowspan=\"2\" style=\"text-align:center\"> Example Words In this Relation </th><tr></tr>'
                       f'</tr>')
            for (relation, _) in sorted_relations:

                relation_string, link = get_relation(relation)
                outp.write(f'<tr><td> <a  href=\"./{relation}_relations.html\">{relation_string} ({relation})</a> </td>')

                word_examples = sorted(deprel_words_examples[relation].items(), key=lambda kv:kv[1], reverse=True)[:5]
                examples_text = "<br> ".join([word for (word, _) in word_examples ])
                outp.write(f'<td>{examples_text}</td></tr>')

                relfilename = f"{helper_file_dir}/{relation}_relations.html"
                with open(relfilename, 'w') as reloutp:
                    HEADER = ORIG_HEADER.replace("main.css", "../../main.css")
                    reloutp.write(HEADER + '\n')
                    reloutp.write(f'<ul class="nav"><li class="nav"><a class="active" href=\"../../index.html\">Home</a>'
                               f'</li><li class="nav"><a href=\"../../introduction.html\">Usage</a></li>'
                               f'<li class="nav"><a href="../../about.html\">About Us</a></li></ul>\n')
                    reloutp.write(
                        f"<br><li><a href=\"./relations.html\">Back to relations page</a></li>\n")


                    reloutp.write(f'<h1> Example constructions for relation <b>{relation} ({relation_string})</b>, the words participating in this relation are marked by ***')
                    reloutp.write(f'<h1> For detailed definition of each relation check <a href="{link}"> here </a>. </h1>\n')
                    for (word, _) in word_examples:
                        sent_examples = deprel_examples[relation][word][:5]
                        prev_filename = None
                        for (filename, sent_num, token_num) in sent_examples:
                            if prev_filename != filename:
                                train_data = pyconll.load_from_file(filename)
                            prev_filename = filename

                            sent = train_data[sent_num]
                            token_ = sent[token_num]
                            headid = token_.head
                            reloutp.write('<pre><code class="language-conllu">\n')
                            for token in sent:
                                transliterated = None
                                if trn:
                                    transliterated = trn.transform(token.form)
                                if token.id == token_.id:
                                    temp = token.conll().split('\t')
                                    if transliterated:
                                        temp[1] = "***" + temp[1] + '(' + transliterated + ')' + "***"
                                    else:
                                        temp[1] = "***" + temp[1]  + "***"
                                    temp2 = '\t'.join(temp)
                                    reloutp.write(temp2 + "\n")
                                elif token.id == headid:
                                    temp = token.conll().split('\t')
                                    if transliterated:
                                        temp[1] = "***" + temp[1] + '(' + transliterated + ')' + "***"
                                    else:
                                        temp[1] = "***" + temp[1] + "***"
                                    temp2 = '\t'.join(temp)
                                    reloutp.write(temp2 + "\n")
                                else:
                                    temp = token.conll().split('\t')
                                    if transliterated:
                                        temp[1] = temp[1] + '(' + transliterated + ')'
                                    else:
                                        temp[1] = temp[1]

                                    temp[6], temp[7], temp[8], temp[9] = '0', "_", "_", "_"
                                    temp2 = '\t'.join(temp)
                                    reloutp.write(temp2 + "\n")

                            reloutp.write('\n</code></pre>\n\n')

                    reloutp.write(FOOTER +"\n")

            outp.write("</table>")
            outp.write(FOOTER + "\n")
        fileoutp.write(FOOTER + "\n")