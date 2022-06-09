import argparse, pyconll, random

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="mr_pan-ud-train.conllu")
parser.add_argument("--output", type=str, default="mr_pan-lem-train.txt")
args = parser.parse_args()

#Treebanks obtained from Paninian treebank have suffix in-place of deprel
if __name__ == "__main__":
    d = pyconll.load_from_file(args.input)
    inflections_different, inflections_same = set(), set()
    with open(args.output, 'w') as fout:
        for sent in d:
            for token in sent:
                suffix = token.deprel
                token_form = token.form
                token_lemma = token.lemma
                token_upos = token.upos


                feats_string = []
                keys = list(token.feats.keys())
                keys.sort()
                for feat in keys:
                    values = list(token.feats[feat])
                    values.sort()
                    value = "/".join(values)
                    feats_string.append(value)

                feats_string = [token_upos] + feats_string
                if suffix == None:
                    inflections_same.add((token_form, token_lemma, ",".join(feats_string), "_"))
                else:
                    inflections_different.add((token_form, token_lemma, ",".join(feats_string), suffix))

        all_data = list(inflections_different) + list(inflections_same)
        random.shuffle(all_data)

        # for (token_form, token_lemma, feats, suffix) in all_data:
        #     if suffix == '0':
        #         suffix = '_'
        #     fout.write(f'{token_form}\t{token_lemma}+{suffix}\t{feats}\n')
        print(len(inflections_different), len(inflections_same))
