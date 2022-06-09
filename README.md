### Teacher Perception ###


## Online interface to Explore Rules
The extracted rules can be explored [here](http://autolex.co/interface). We cover the following grammar aspects in the tool.

1. *General Information* ([here](https://aditi138.github.io/auto-lex-learn-general-info/mr_en/helper/syntactic_info.html))
2. *Word Usage*
    - Semantic Subdivisions ([here](https://aditi138.github.io/auto-lex-learn-word-usage/mr_en/WordUsage/WordUsage.html))
    - Basic Words ([here](https://aditi138.github.io/auto-lex-learn/mr_en/WordUsage/vocab.html))
    - Adjectives, Synonyms, Antonyms ([here](https://aditi138.github.io/auto-lex-learn/mr_en/WordUsage/vocab_adj.html))
3. *Word Order* ([here](http://www.autolex.co/interface/es_gsd/WordOrder/WordOrder.html))
4. *Suffix Usage* ([here](http://www.autolex.co/interface/mr_en/Suffix/Suffix.html))
5. *Agreement* ([here](http://www.autolex.co/interface/mr_en/Agreement/Agreement.html))

## Code for Extracting the Learning Materials
All the required scripts are `code/`, to view the rules in the required website, first create a folder (`website`) to hold the rules and copy the source html files --
```
   mkdir -p website
   cp html_pages/* website
```

**For word order**:
To extract word order patterns, run --
```
    cd code/
    python word_order.py \
    --input ../data/ \
    --file input_files.txt \
    --lexical \
    --use_xgboost \
    --best_depth 10 10 10 10 10 \
    --folder_name website/
    --transliterate mar
```
`--best_depth` sets the depth of the decision tree, generally 10 is a good number to extract rules that result in a reasonable model performance with keeping them concise.
But if you want to find the best performing depth, we recommend the `--no_print` option, which would only print the model accuracy with different depths, but this will not
print the rules in the website, after finding the best depth, you could run the above command by removing the `--no_print` option.
The above command extract rules for five orders: subject-verb, object-verb, adjective-noun, numeral-noun and adposition-noun. The depth parameters should also be specified
in that order for the required model. If you don't want to run models for all orders, specify which orders to skip in the option like this `--skip_models subject-verb object-verb`
and in the best-depth field simply specify -1 (e.g. `--best_depth -1 -1 10 10 10`)/
`--transliterate` option will allow you to transliterate the examples in the Roman script, note this option is only available for Marathi (mar) and Kannada (kan) languages,
for other languages simply do not add it.

**For agreement**:
```
    cd code/
    python agreement_per_pos.py \
    --input ../data/ \
    --file input_files.txt \
    --lexical \
    --use_spine \
    --use_xgboost \
    --best_depth 10 10 10 \
    --features Gender+Person+Number \
    --folder_name website/
    --transliterate mar
```

**For general information**:
```
    cd code/
    python general_information.py \
    --input ../data/SUD_Marathi-Sample/mr_sam-sud-train.conllu \
    --folder_name website \
    --lang mr \
    --transliterate mar \
```
Running this command will extract answer descriptions for understanding syntax of the language, for example, how many genders this language has, which words take what genders, and so on.
This will also extract illustrative examples for each word.
If the input treebank is large (i.e > 20k sentences), we recommend breaking the file into smaller files and using the option `--distributed_files <path to the smaller files>`
It may still take a long time to process with many files as it aggregates the information.

**For word usage**
This requires a parallel corpus with [word alignments](https://github.com/neulab/awesome-align) already run, a sample is provided in the data folder. We break the code into three components:
To get popular adjectives, synonyms and antonyms,
```
    cd code/
    python vocabulary_adj.py \
    --en_input ../data/SUD_English-Sample/en_sam-sud-train.conllu \
    --mr_input ../data/SUD_Marathi-Sample/mr_sam-sud-train.conllu \
    --alignment ../data/alignments/train.pred \
    --input ../data/en_mr_vocab.txt \
    --en_adj ../data/en_adj.txt \
    --folder_name website \
    --lang mr \
    --transliterate mar
```
The script first extracts popular adjectives and vocabulary from the data, and stores them in locations specified by `--en_adj` and `--input`.
If these files are already existing, they won't be overwritten.

To get words for popular semantic categories, first we need to identify the wordnet senses for each English word in our corpus.
to identify wordnets we used the [XL-WSD](https://github.com/SapienzaNLP/xl-wsd-code).
```
    cd code/
    python mapBabelIDSUD.py \
    --en_input ../data/SUD_English-Sample/en_sam-sud-train.conllu \
    --mr_input ../data/SUD_Marathi-Sample/mr_sam-sud-train.conllu \
    --alignment ../data/alignments/train.pred \
    --wsd_output_pred ~/xl-wsd-code/data/output/models/roberta-large_xlm-roberta-large/evaluation/wsd-en-mr.predictions.txt \
    --wsd babelid_synset.txt \
    --output vocab_wordnet.txt
```
A sample output from the XL-WSD is shown in `data/wsd-en-mr.predictions.txt', please refer to the XL-WSD code to know how to format the data (in our case the English raw sentences) appropirately.
The XL-WSD code outputs the babelids for each word in the sentence (e.g. 7000.1 denotes the 1st token in 7000th input sent).
To convert the babelid into the wordnet sense, we use the [BabelNet API](https://babelnet.io/v6/getOutgoingEdges), example output is shown in `data/babelid_synset.txt`.
This will output the `vocab_wordnet.txt` which contains the wordnet id with corresponding English words (example in `data/vocab_wordnet.txt`).
Next, to identify the words in the target languge with examples run--
```
    cd code/
    python vocabulary_basic_words.py \
    --input data/vocab_wordnet.txt \
    --folder_name website \
    --lang mr \
    --transliterate mar \
    --en_adj en_adj.txt
```
`en_adj.txt` was created in the script `vocabulary_adj.py`.

To get the semantic subdivisions, refer [here](https://github.com/Aditi138/LexSelection).

**For suffix usage**
The first thing we need is to identify all the suffixes in a word with its morphological properties. For Marathi and Kannada, the treebanks we release (mentioned below) have suffixes annotated.
We use that as a training data to identify suffixes in our large corpora.
Specifically, we use the model ([here](https://github.com/tatyana-ruzsics/interpretable-inflection)) to transform the suffix data in the expected format.
The original model expects a lemma with its morphological analyses and outputs the inflected form.
We modify the input/output format to take input as the inflected word form with its morphological analyses and produce as output the lemma with each suffix.
Sample training data for that is shown in `data/train_lem.txt`, for creating this data from the treebanks run--
```
    cd code/
    python helper_scripts/createLemmaTraining.py --input train.conllu --dev train_lem.txt
```
Setup the code from [here](https://github.com/Aditi138/interpretable-inflection) and run--
```
    $DIR=~//interpretable-inflection
    cd $DIR
    $train=~/parallel-data/en-mr/mr_pan-train-lem.txt
    $dev=~/parallel-data/en-mr/mr_pan-dev-lem.txt
    ./train-ch.sh mr_models $train $dev config/gate-sparse-enc-static-head.yml # chED+chSELF-ATT model
```
The models learnt for Marathi and Kannada are stored in `mr_models.zip` and `kn_models.zip`, which can be accessed [here](https://www.autolex.co/download.html)
You can use the `predict_kn.sh` example script to use the trained models to predict on similar formatted data.
Since, we had use the [Samanantar dataset](https://indicnlp.ai4bharat.org/samanantar/) which comprises of 4M sentences, we segmented the data into smaller files with 20k sents.
The results from all predictions (which are included in the `mr_models/predict/`) are aggregated using  `run.sh`, which identifies the top-20 popular suffixes and the learnt segmentation for individual words, the ones we used are stored in `outputs.zip` ([here](https://www.autolex.co/download.html)).


### Data for training Kannada and Marathi Syntactic Parser
The treebanks converted in the SUD format can be found [here](https://github.com/Aditi138/auto-lex-learn/tree/master/data).
It contains POS tags, lemmatization, morphological analyses and dependency analyses, note these are automatically converted data and not manually verified.
