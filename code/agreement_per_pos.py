import  os, pyconll
import  dataloader_agreement_per_pos as dataloader
import argparse
import numpy as np

np.random.seed(1)
import sklearn
from collections import defaultdict
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import utils
from scipy.sparse import csr_matrix, vstack
from scipy.sparse import csc_matrix
import xgboost as xgb
from io import  StringIO

DTREE_DEPTH = [3, 4, 5, 6, 7, 8, 9, 10]
DTREE_IMPURITY = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
XGBOOST_GAMMA = [0.005, 0.01, 1, 2, 5, 10]

def printTreeFromXGBoost(model, rel, folder_name, train_data_path, train_path, test_data_path, test_path, lang, tree_features, relation_map, genreparseddata, wikidata, spine_word_vectors, spine_features, spine_dim):
    best_model = model.best_estimator_
    rules_df = best_model.get_booster().trees_to_dataframe()
    rules_df_t0 = rules_df.loc[rules_df['Tree'] == 0]
    # Iterate the tree to get information about the tree features
    topnodes, tree_dictionary, leafnodes, leafedges, edge_mapping = utils.iterateTreesFromXGBoost(rules_df_t0, args.task, f'{args.features}-{rel}',
                                                                                       relation_map, tree_features, folder_name, args.transliterate)

    # Assign statistical threshold to each leaf
    train_df = pd.read_csv(train_data_path, sep=',')
    train_data = pyconll.load_from_file(train_path)
    leaf_examples, leaf_sent_examples = utils.FixLabelTreeFromXGBoostWithFeatures(rules_df_t0, tree_dictionary, leafnodes,
                                                                      label_list, data_loader.model_data_case[rel],
                                                                      f'{args.features}-{rel}', train_df, train_data,
                                                                      tree_features, data_loader, genreparseddata,
                                                                      wikidata, spine_word_vectors, spine_features, spine_dim,
                                                                      task=args.task)

    tree_dictionary, leafmap = utils.collateTreeFromXGBoost(tree_dictionary, topnodes, leafnodes, leafedges)

    test_df = pd.read_csv(test_data_path, sep=',')
    test_data = pyconll.load_from_file(test_path)
    metric, _ = utils.computeAutomatedMetric(leafmap, tree_dictionary, test_df, test_data, f'{args.features}-{rel}', tree_features, data_loader, args.task, genreparseddata, wikidata, spine_word_vectors,
                                       spine_features, spine_dim, folder_name, lang)
    baselinemetric, _ = utils.computeAutomatedMetric(leafmap, tree_dictionary, test_df, test_data, f'{args.features}-{rel}', tree_features, data_loader, args.task, genreparseddata, wikidata, spine_word_vectors,
                                       spine_features, spine_dim, folder_name, lang, isbaseline=True)

    print("Model: %s-%s, Lang: %s, Test Accuracy : %.4g, Baseline Accuracy: %.4g"  % (args.features, rel, lang_full, metric * 100.0, baselinemetric * 100.0))
    print(f"Hyperparameter tuning: ", model.best_params_)
    print("Num of leaves", len(leafmap))

    if args.no_print:
        return tree_dictionary, leafmap, {}, {}, {}


    retainNA = args.retainNA
    leaf_values_per_columns, columns_per_leaf, update_spine_features = {}, {}, {}

    important_features, cols = utils.getImportantFeatures(tree_dictionary, leafmap, "", retainNA=retainNA)
    if len(important_features) == 0:  # If there is no sig other label
        return tree_dictionary, leafmap, {}, {}, {}

    leaf_examples_features = {}
    total_num_examples = 0
    num_examples_label = defaultdict(lambda: 0)
    num_leaf_examples = {}

    with open(f'{folder_name}/{lang}_{rel}_rules.txt', 'w') as fleaf:
        for leaf_num in range(len(leafmap)):
            leaf_examples_features[leaf_num] = {}
            leaf_index = leafmap[leaf_num]
            columns_per_leaf[leaf_num] = defaultdict(list)
            leaf_label = tree_dictionary[leaf_index]['class']
            # if leaf_label == 'NA':
            #     leaf_label = 'chance-agree'
            #     tree_dictionary[leaf_index]['class'] = leaf_label
            leaf_examples_features[leaf_num]['leaf_label'] = leaf_label

            examples = leaf_examples[leaf_index]
            sent_examples = leaf_sent_examples[leaf_index]

            agree, disagree, total_agree, total_disagree, spine_features = utils.getExamples(sent_examples,
                                                                                             tree_dictionary[
                                                                                                 leaf_index],
                                                                                             train_data, isTrain=True)
            for keyspine, valuespines in spine_features.items():
                key = keyspine.split("_")[0]
                sorted_spine = sorted(valuespines.items(), key=lambda kv:kv[1], reverse=True)[:10]
                values = [v for (v,_) in sorted_spine]
                update_spine_features[keyspine] = f'{key}_{",".join(values)}'
            leaf_examples_features[leaf_num]['agree'] = agree
            leaf_examples_features[leaf_num]['disagree'] = disagree



            row = f'<tr>'

            active_all = tree_dictionary[leaf_index]['active'] #features which are active for this leaf
            non_active_all = tree_dictionary[leaf_index]['non_active'] #features which are not active for this leaf
            top = tree_dictionary[leaf_index].get('top', -1)

            while top > 0:  # Not root
                active_all += tree_dictionary[top]['active']
                non_active_all += tree_dictionary[top]['non_active']
                top = tree_dictionary[top]['top']

            active, non_active = [],[]
            for (feat, _) in active_all:
                feat = update_spine_features.get(feat, feat)
                active.append(feat)
            for (feat, _) in non_active_all:
                feat = update_spine_features.get(feat, feat)
                non_active.append(feat)

            active, non_active = list(set(active)), list(set(non_active))
            leaf_examples_features[leaf_num]['active'] = active
            leaf_examples_features[leaf_num]['non_active'] = non_active

            fleaf.write(leaf_label + "\n")
            fleaf.write('active: ' + " ### ".join(active) + "\n")
            fleaf.write('non_active: ' + " ### ".join(non_active) + "\n")
            fleaf.write(
                f'agree:{total_agree}, disagree: {total_disagree}, total: {total_agree + total_disagree}\n')
            fleaf.write("\n")

            # populating only rules which deviate from the dominant label as observed from the training data
            for feat in important_features:
                feat = update_spine_features.get(feat, feat)
                if feat in active:

                    columns_per_leaf[leaf_num]['Y'].append(feat)
                    row += f'<td style=\"text-align:center\"> Y </td>'
                elif feat in non_active:

                    columns_per_leaf[leaf_num]['N'].append(feat)
                    row += f'<td style=\"text-align:center\"> N </td>'
                else:

                    columns_per_leaf[leaf_num]['-'].append(feat)
                    row += f'<td style=\"text-align:center\"> - </td>'
            row += f'<td> {leaf_label} </td> <td> <a href=\"Leaf-{leaf_num}.html\"> Examples</a> </td>\n'
            leaf_examples_features[leaf_num]['row'] = row


            #get the number of examples predicted for each label
            for _, num_example_per_leaf in sent_examples.items():
                num_examples_label[leaf_label] += len(num_example_per_leaf)
                total_num_examples += len(num_example_per_leaf)

            num_leaf_examples[leaf_num] = total_agree + total_disagree


    return tree_dictionary, leafmap, leaf_examples_features, num_leaf_examples, columns_per_leaf

def dtreemodel(feature, x,y, x_test, y_test, impurity,cv):
    criterion = ['gini', 'entropy']

    #parameters = {'criterion': criterion, 'max_depth': depth, 'min_impurity_decrease': [1e-3, 1e-1, 1e-2]}
    parameters = {'criterion': criterion, 'min_impurity_decrease': impurity, 'min_samples_split': [0.005]}
    decision_tree = DecisionTreeClassifier()

    model = GridSearchCV(decision_tree, parameters, cv=cv)
    model.fit(x, y)

    return model

def xgboostModel(rel, x,y, x_test, y_test, depth,cv):
    criterion = ['gini', 'entropy']
    parameters = {'criterion': criterion, 'max_depth': depth, 'min_child_weight': [10]}

    xgb_model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=1, subsample=0.8,
                      colsample_bytree=0.8, objective='multi:softprob', num_class=len(label_list), silent=True, nthread=args.n_thread,
                      scale_pos_weight=majority / minority, seed=1001)

    model = GridSearchCV(xgb_model, parameters, cv=cv, n_jobs=args.n_jobs)
    #sparse_x = csr_matrix(x)
    model.fit(x, y)
    return model

def train(train_features, train_labels, train_path, train_data_path,
          dev_features, dev_labels,
          test_features, test_labels, test_path, test_data_path,
          rel, baseline_label,
          folder_name, lang_full, tree_features_dict,
          relation_map, best_depths,
          genreparseddata, wikidata, spine_word_vectors, spine_features, spine_dim, test_datasets):
    x_train, y_train, x_test, y_test = train_features, train_labels, test_features, test_labels

    tree_features = []
    for f in range(len(tree_features_dict)):
        tree_features.append(tree_features_dict[f])
    tree_features = np.array(tree_features)
    if dev_features is not None:
        x_dev, y_dev = dev_features, dev_labels
        x = vstack((x_train, x_dev))
        y = np.concatenate([y_train, y_dev])
        test_fold = np.concatenate([
            # The training data.
            np.full(x_train.shape[0], -1, dtype=np.int8),
            # The development data.
            np.zeros(x_dev.shape[0], dtype=np.int8)
        ])
        cv = sklearn.model_selection.PredefinedSplit(test_fold)
    else:
        x, y = x_train, y_train
        cv = 5
        if args.use_xgboost:
            x = x.to_numpy()

    # Print acc/leaves for diff settings
    if args.no_print:
        for depth in DTREE_DEPTH:  #:
            setting = f'-g-{depth}'
            model = xgboostModel(args.features, x, y, x_test, y_test, [depth], cv)
            best_model = model.best_estimator_
            best_model.get_booster().dump_model(f'{folder_name}/{rel}/xgb_model-{setting}.txt', with_stats=True)
            tree_dictionary, leafmap, leaf_examples_features, num_leaf_examples, columns_per_leaf = printTreeFromXGBoost(model, rel, folder_name, train_data_path, train_path, test_data_path, test_path, lang, tree_features,  relation_map, genreparseddata, wikidata, spine_word_vectors, spine_features, spine_dim)
        print()

    else:
        if args.features in best_depths and best_depths[args.features] != -1:
            model = xgboostModel(args.features, x, y, x_test, y_test, [best_depths[args.features]], cv)
            best_model = model.best_estimator_
            best_model.get_booster().dump_model(f'{folder_name}/{rel}/xgb_model.txt', with_stats=True)
            tree_dictionary, leafmap, leaf_examples_features, num_leaf_examples, columns_per_leaf = printTreeFromXGBoost(model, rel, folder_name, train_data_path, train_path, test_data_path, test_path, lang, tree_features,  relation_map, genreparseddata, wikidata, spine_word_vectors, spine_features, spine_dim)

        else:
            # model = dtreemodel(args.features, x, y, x_test, y_test, DTREE_DEPTH, cv)
            # dot_data = StringIO()
            # label_list_string = ['NA', 'req-agree']
            # sklearn.tree.export_graphviz(model.best_estimator_, out_file=dot_data,
            #                              feature_names=tree_features, node_ids=True
            #                              , class_names=label_list_string, proportion=False, rounded=True, filled=True,
            #                              leaves_parallel=False, impurity=False)

            model = xgboostModel(args.features, x, y, x_test, y_test, DTREE_DEPTH, cv)
            best_model = model.best_estimator_
            best_model.get_booster().dump_model(f'{folder_name}/{rel}/xgb_model.txt', with_stats=True)
            tree_dictionary, leafmap, leaf_examples_features, num_leaf_examples, columns_per_leaf = printTreeFromXGBoost(
                model, rel, folder_name, train_data_path, train_path, test_data_path, test_path, lang, tree_features,
                relation_map, genreparseddata, wikidata, spine_word_vectors, spine_features, spine_dim)

    best_model = model.best_estimator_
    with open(f"{folder_name}/{rel}/{rel}.html", 'w') as output:
        # Get the features in breadth first order of the class which deviates from the dominant order
        HEADER = ORIG_HEADER.replace("main.css", "../../../../main.css")
        output.write(HEADER + '\n')
        output.write(f'<ul class="nav"><li class="nav"><a class="active" href=\"../../../../index.html\">Home</a>'
                   f'</li><li class="nav"><a href=\"../../../../introduction.html\">Usage</a></li>'
                   f'<li class="nav"><a href="../../../../about.html\">About Us</a></li></ul>')
        output.write(f"<br><li><a href=\"../../Agreement.html\">Back to {language_fullname} page</a></li>\n")

        output.write(f"<h1> Rules for {args.features} agreement for <b>{rel}</b> </h1>")
        output.write(
            f'<h2> The {args.features} values <b> should match </b> between the <b> {rel} </b>  and its governor (i.e syntactic head)  when <b>label = should-match</b>,\n '
            f'else any observed agreement is purely by chance (<b>label = need-not-match)</b> </h2> ')


        if baseline_label == 0:
            dominant_label = 'chance-agree'
        else:
            dominant_label = 'req-agree'

        train_data = pyconll.load_from_file(train_path)
        if not args.no_print:
            # Sort leaves by number of examples, and remove the col from first general leaf
            sorted_leaves = sorted(num_leaf_examples.items(), key=lambda kv: kv[1], reverse=True)
            first_leaf, columns_to_remove, columns_to_retain = True, set(), set()
            first_leaf, retainNA = True, args.retainNA

            output.write(
                f'<table>'
                f'<tr><th colspan=\"rowspan=\"3\"	scope=\"colgroup\" \" style=\"text-align:center\"> Agreement </th>'
                f'<th colspan=\"rowspan=\"2\"	scope=\"colgroup\" \" style=\"text-align:center\"><b>  </th>'
                '</tr><tr>\n')

            active_features_per_nondomleaf, columns_with_activeinactive_features, column_with_leafinfo, columns_active_indomleaves, columns_in_innondomleaves = [], {}, {}, set(), set()
            for (leaf_num, _) in sorted_leaves:
                leaf_label = leaf_examples_features[leaf_num]['leaf_label']
                if not retainNA and leaf_label == 'NA':
                    continue

                if first_leaf and leaf_label == dominant_label:  # This is the leaf which says the general word-order
                    leaf_examples_features[leaf_num]['cols'] = ['generally between the head and depenent there is ']
                    leaf_examples_features[leaf_num]['default'] = True
                    dominant_label = leaf_label
                    first_leaf = False

                active_columns_in_leaf = columns_per_leaf[leaf_num].get('Y', [])
                nonactive_columns_in_leaf = columns_per_leaf[leaf_num].get('N', [])
                missing_columns_in_leaf = columns_per_leaf[leaf_num].get('-', [])

                if leaf_label != dominant_label and len(
                        active_columns_in_leaf) > 0:  # There is at least one active feature in the leaf
                    active_features_per_nondomleaf += active_columns_in_leaf + nonactive_columns_in_leaf

                all_cols = set(active_columns_in_leaf + nonactive_columns_in_leaf + missing_columns_in_leaf)
                for c in all_cols:
                    if c in active_features_per_nondomleaf:
                        if leaf_label != dominant_label: #feature is present for a non-dominant leaf
                            columns_in_innondomleaves.add(c)

                        columns_to_retain.add(c)
                        if c not in columns_with_activeinactive_features:
                            columns_with_activeinactive_features[c] = defaultdict(lambda: 0)

                        if c in active_columns_in_leaf:
                            columns_with_activeinactive_features[c]['Y'] += 1
                            if leaf_label == dominant_label:
                                columns_active_indomleaves.add(c)

                        elif c in nonactive_columns_in_leaf or c in missing_columns_in_leaf:
                            columns_with_activeinactive_features[c]['N'] += 1

            #If a feature is not active for any leaf. but is present for a non-dominant leaf, we keep it else remove it
            for c, leafdata in columns_with_activeinactive_features.items():
                leaves_with_active_column = leafdata.get('Y', 0)
                if leaves_with_active_column == 0 and c not in columns_in_innondomleaves:
                    columns_to_retain.remove(c)

            for c in columns_active_indomleaves:  # for col active in dominant leaves, check if its active (Y,N) in non-dom leaves
                to_keep = False
                for (leaf_num, _) in sorted_leaves:
                    leaf_label = leaf_examples_features[leaf_num]['leaf_label']
                    if not retainNA and leaf_label == 'NA':
                        continue

                    if leaf_label != dominant_label:
                        active_columns_in_leaf = columns_per_leaf[leaf_num].get('Y', [])
                        nonactive_columns_in_leaf = columns_per_leaf[leaf_num].get('N', [])

                        if c in active_columns_in_leaf or c in nonactive_columns_in_leaf:
                            to_keep = True

                    if to_keep:
                        break

                if not to_keep:
                    columns_to_retain.remove(c)

            # Create the table of features
            columns_to_retain = list(columns_to_retain)
            if len(columns_to_retain) == 0 and not args.no_print:  # all NA leaves:
                return

            col_names = ["" for _ in range(len(columns_to_retain))]
            cols = utils.getColsToCombine(columns_to_retain)
            for combined in cols:
                header, subheader = utils.getHeader(combined, columns_to_retain, f'{args.features}-{rel}', args.task, relation_map, folder_name, args.transliterate)
                #output.write(
                #    f'<th colspan=\"{len(combined)}	scope=\"colgroup\" \" style=\"text-align:center;width:130px\">{header}</th>\n')
                for col, header in zip(combined, subheader):
                    col_names[col] = header


            first_leaf, leaves_covered = True, set()
            retainNA = args.retainNA
            non_dominant_leaves = defaultdict(list)
            with open(f'{folder_name}/{lang}_{rel}_printable_rules.txt', 'w') as fleaf:
                for (leaf_num, _) in sorted_leaves:
                    leaf_label = leaf_examples_features[leaf_num]['leaf_label']
                    if not retainNA and leaf_label == 'NA':
                        continue

                    if first_leaf:
                        leaf_examples_file = f'{folder_name}/{rel}/Leaf-{leaf_num}.html'
                        active, non_active = leaf_examples_features[leaf_num]['active'], leaf_examples_features[leaf_num][
                            'non_active']
                        agree, disagree = leaf_examples_features[leaf_num]['agree'], leaf_examples_features[leaf_num][
                            'disagree']

                        yes_cols, no_cols, missin_cols = columns_per_leaf[leaf_num].get('Y', []), columns_per_leaf[
                            leaf_num].get('N', []), columns_per_leaf[leaf_num].get('-', [])
                        num_features_in_leaf, cols, yes_cols_number, no_cols_number, missing_cols_number = 0, [], 0, 0, 0
                        yes_cols_text, no_cols_text = [], []
                        row = '<tr>'
                        for column in columns_to_retain:  # Iterate all columns
                            if column in yes_cols:
                                row += f'<td style=\"text-align:center\"> Y </td>'
                                num_features_in_leaf += 1
                                yes_cols_number += 1
                                cols.append(f'Y_{column}')
                                yes_cols_text.append(column)
                            elif column in no_cols:
                                row += f'<td style=\"text-align:center\"> N </td>'
                                num_features_in_leaf += 1
                                no_cols_number += 1
                                cols.append(f'N_{column}')
                                no_cols_text.append(column)
                            else:
                                row += f'<td style=\"text-align:center\"> - </td>'
                                cols.append(f'-_{column}')
                                missing_cols_number += 1

                        if leaf_examples_features[leaf_num].get('default', False):
                            row = f'<tr> <th colspan=\"{len(columns_to_retain)}	scope=\"colgroup\" \" style=\"text-align:center;width:130px\">generally for most head-dependent pairs there is </th>\n'
                            row += f'<td> {leaf_label} </td> <td> <a href=\"Leaf-{leaf_num}.html\"> Examples</a> </td>'
                            utils.populateLeaf(train_data, leaf_examples_file, leaf_label, agree, disagree,
                                               active, non_active, first_leaf, f'{args.features}-{rel}', args.task, lang_full, relation_map,
                                               columns_to_retain, ORIG_HEADER, HEADER,
                                               FOOTER, folder_name, args.transliterate)
                            cols = ['default']
                            leaf_examples_features[leaf_num]['valid'] = True
                            fleaf.write(f'default\t\t{leaf_label}\n')
                            #fleaf.write(f'{"~~~".join(yes_cols_text)}\t{"~~~".join(no_cols_text)}\t{leaf_label}\n')

                            if leaf_label == 'req-agree':
                                label_str = 'should match'
                            else:
                                label_str = 'need not match'
                            output.write(
                                f'<tr> <td style=\"text-align:center\"> Generally <b>{args.features} {label_str}<b> between the {rel} and its governor or head  </td> </tr>\n')
                            output.write(
                                f'<tr><td style=\"text-align:center\"> Some examples are: <a href="Leaf-{leaf_num}.html"> Examples </a> </td></tr>\n')


                        else:
                            if yes_cols_number > 0:
                                leaf_examples_features[leaf_num]['valid'] = True
                                utils.populateLeaf(train_data, leaf_examples_file, leaf_label, agree, disagree,
                                                   active, non_active, False, f'{args.features}-{rel}', args.task, lang_full, relation_map,
                                                   columns_to_retain, ORIG_HEADER, HEADER,
                                                   FOOTER, folder_name, args.transliterate)
                                fleaf.write(f'{"~~~".join(yes_cols_text)}\t{"~~~".join(no_cols_text)}\t{leaf_label}\n')
                                feature_def = ""
                                for feat_name in yes_cols_text:
                                    feature_def += utils.transformRulesIntoReadable(feat_name, args.task, f'{args.features}-{rel}', relation_map, f'{args.folder_name}/{lang}', data_loader.examples_per_pos, args.transliterate) + '\n<br>'

                                non_dominant_leaves[leaf_label].append((leaf_num, feature_def))
                                # output.write(
                                #     f'<tr> <td style=\"text-align:center\"> <b> {args.features} {label_str} </b> between the {rel} and its governor or head when:  </td></tr>')
                                # output.write(
                                #     f'<tr><td style=\"text-align:center\"> {feature_def} </td> <td> <a href="Leaf-{leaf_num}.html"> Examples </a> </td></tr>')

                            else:
                                leaf_examples_features[leaf_num]['valid'] = False
                            row += f'<td> {leaf_label} </td> <td> <a href=\"Leaf-{leaf_num}.html\"> Examples</a> </td>'

                        leaf_examples_features[leaf_num]['row'] = row
                        first_leaf = False
                        leaf_examples_features[leaf_num]['dispcols'] = cols
                        cols.sort()
                        leaf_cols = f'{leaf_label}{";".join(list(cols))}'
                        leaves_covered.add(leaf_cols)
                    else:

                        yes_cols, no_cols, missin_cols = columns_per_leaf[leaf_num].get('Y', []),  columns_per_leaf[leaf_num].get('N', []), columns_per_leaf[leaf_num].get('-', [])
                        row = '<tr>'
                        num_features_in_leaf, cols, yes_cols_number, no_cols_number, missing_cols_number = 0, [], 0, 0, 0
                        yes_cols_text, no_cols_text = [], []
                        for column in columns_to_retain:  # Iterate all columns
                            if column in yes_cols:
                                row += f'<td style=\"text-align:center\"> Y </td>'
                                num_features_in_leaf += 1
                                yes_cols_number += 1
                                cols.append(f'Y_{column}')
                                yes_cols_text.append(column)
                            elif column in no_cols:
                                row += f'<td style=\"text-align:center\"> N </td>'
                                num_features_in_leaf += 1
                                no_cols_number += 1
                                cols.append(f'N_{column}')
                                no_cols_text.append(column)
                            else:
                                row += f'<td style=\"text-align:center\"> - </td>'
                                cols.append(f'-_{column}')
                                missing_cols_number += 1

                        cols.sort()
                        leaf_cols = f'{leaf_label}{";".join(list(cols))}'
                        if leaf_cols in leaves_covered or yes_cols_number == 0:  # remove leaves which have all features 'N' or "-"
                            leaf_examples_features[leaf_num]['valid'] = False
                            continue

                        row += f'<td> {leaf_label} </td> <td> <a href=\"Leaf-{leaf_num}.html\"> Examples</a> </td>\n'
                        leaf_examples_features[leaf_num]['row'] = row
                        leaf_examples_features[leaf_num]['dispcols'] = set(cols)
                        leaf_examples_features[leaf_num]['valid'] = True
                        fleaf.write(f'{"~~~".join(yes_cols_text)}\t{"~~~".join(no_cols_text)}\t{leaf_label}\n')

                        feature_def = ""
                        for feat_name in yes_cols_text:
                            feature_def += utils.transformRulesIntoReadable(feat_name, args.task,
                                                                            f'{args.features}-{rel}', relation_map,
                                                                            f'{args.folder_name}/{lang}',
                                                                            data_loader.examples_per_pos,
                                                                            args.transliterate) + '\n<br>'

                        non_dominant_leaves[leaf_label].append((leaf_num, feature_def))

                        leaf_examples_file = f'{folder_name}/{rel}/Leaf-{leaf_num}.html'
                        active, non_active = leaf_examples_features[leaf_num]['active'], leaf_examples_features[leaf_num]['non_active']
                        agree, disagree = leaf_examples_features[leaf_num]['agree'], leaf_examples_features[leaf_num][
                            'disagree']
                        utils.populateLeaf(train_data, leaf_examples_file, leaf_label, agree, disagree,
                                           active, non_active, first_leaf, f'{args.features}-{rel}', args.task, lang_full, relation_map, columns_to_retain, ORIG_HEADER, HEADER, FOOTER
                                           , folder_name, args.transliterate)
                        leaves_covered.add(leaf_cols)

            first = True
            for leaf_label, leaf_info in non_dominant_leaves.items():
                if leaf_label == 'req-agree':
                    label_str = 'should match'
                else:
                    label_str = 'need not match'
                if leaf_label == dominant_label:
                    if first:
                        output.write(
                            f'<tr> <td style=\"text-align:center\"> Generally <b>{args.features} {label_str}<b> between the {rel} and its governor or head  </td> </tr>\n')
                        output.write(
                                f'<tr><td style=\"text-align:center\"> Some examples are: <a href="Leaf-{leaf_num}.html"> Examples </a> </td></tr>\n')
                        first = False
                    continue

                if leaf_label == 'req-agree':
                    label_str = 'should match'
                else:
                    label_str = 'need not match'
                output.write(
                    f'<tr> <td style=\"text-align:center\"> <b> {args.features} {label_str} </b> between the {rel} and its governor or head when: </td> </tr>')

                output.write(f'<tr> <td style=\"text-align:center\">')
                num = 0
                for (leaf_num, feature_def) in leaf_info:
                    output.write(f'{feature_def} (<a href="Leaf-{leaf_num}.html"> Examples </a>)\n<br>\n')
                    if num < len(leaf_info) - 1:
                        output.write(f'<b>OR </b> <br><br>\n')
                    num += 1
                output.write(f'</tr></td>')
            output.write(f'</table>')
    print()

def plot_coefficients_label(classifier, feature_names, label_list, top_features=5):
    coef = classifier.coef_
    num_classes = coef.shape[0]
    important_features = {}
    if num_classes > 2:
        for class_ in range(num_classes):
            label = label_list[class_]
            print(label)
            coefficients = coef[class_, :]
            top_positive_coefficients = np.argsort(coefficients)[-top_features:]
            top_negative_coefficients = np.argsort(coefficients)[:top_features]
            top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
            feature_names = np.array(feature_names)
            required_features = list(feature_names[top_coefficients][-top_features:])
            required_features.reverse()
            features = []
            for r in required_features:
                info = r

                value = data_loader.feature_freq[pos].get(info,0)
                features.append(r + ", " + str(value))
            print("\n".join(features))
            print()
            important_features[class_] = required_features
    else:
        coef = coef.ravel()
        top_positive_coefficients = np.argsort(coef)[-top_features:]
        top_negative_coefficients = np.argsort(coef)[:top_features]
        top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

        # Class = 0
        label = label_list[0].split("/")[0]
        print(label)
        required_features = list(feature_names[top_negative_coefficients])
        #required_features.reverse()
        features = []
        for r in required_features:
            info = r
            value = data_loader.feature_freq[pos].get(info,0)
            features.append(r + ", " + str(value))
        print("\n".join(features))

        important_features[0] = required_features

        # Class = 1
        label = label_list[1].split("/")[0]
        print(label)
        required_features = list(feature_names[top_positive_coefficients])
        required_features.reverse()
        features = []
        for r in required_features:
            info = r
            value = data_loader.feature_freq[pos].get(info, 0)
            features.append(r + ", " + str(value))
        print("\n".join(features))
        important_features[1] = required_features

    return important_features

def loadDataLoader(vocab_file, data_loader):
    data_loader.feature_map_id2model, label2id,label_list, data_loader.examples_per_pos = {}, {}, [], {}
    with open(vocab_file, 'r') as fin:
        for line in fin.readlines():
            if line == "" or line == '\n':
                continue
            elif line.startswith("Model:"):
                rel = line.strip().split("Model:")[-1]
                data_loader.feature_map[rel] = {}
                data_loader.model_data_case[rel] = {}
                data_loader.examples_per_pos[rel] = []

            elif line.startswith("Data:"):
                info = line.strip().split("\t")
                num = int(info[0].split("Data:")[-1])
                data_loader.model_data[rel] = int(info[0].split("Data:")[-1])
                if num == 0:
                    continue
                labels_values = info[2].split(";")
                for label in labels_values:
                    lv = label.split(",")
                    data_loader.model_data_case[rel][lv[0]] = int(lv[1])

            elif line.startswith("Feature"):
                info = line.strip().split("\t")
                feature_id, feature_name = info[0].lstrip().rstrip(), info[1].lstrip().rstrip()
                data_loader.feature_map[rel][feature_name] = int(feature_id.split("Feature:")[1])

            elif line.startswith("Labels"):
                info = line.strip().split("Labels:")[1].split(",")
                label_list = []
                for l in info:
                    label_id, label_name = l.split("-")[0], l.split("-")[1]
                    label_list.append(label_id)
                    label2id[label_name] = int(label_id) #0-0, 1-1


            elif line.startswith('POS:'):
                info = line.strip().split("POS:")[1]
                data_loader.examples_per_pos[rel].append(info)

    for rel, _ in data_loader.feature_map.items():
        data_loader.feature_map_id2model[rel] = {v: k for k, v in data_loader.feature_map[rel].items()}
    print(f'Loaded the vocab from {vocab_file}')
    data_loader.lang_full = train_path.strip().split('/')[-2].split('-')[0][3:]
    return label2id, label_list

def filterData(data_loader):
    """ Only train models for those relations/POS which have at least 50 examples for every label"""
    classes = {}
    for model, count in data_loader.model_data.items():
        if count < 100: #Overall number of examples for that model is less than 100
            continue

        labels = data_loader.model_data_case[model]
        if len(labels) == 1:
            print(f'{model} always has  {labels[0]}  syntactic head')
            continue

        to_retain_labels = []
        for label, value in labels.items():
            if value < 50:
                continue
            to_retain_labels.append((label, value))

        if len(to_retain_labels) == 1:
            print(f'{model} majority of the is  {to_retain_labels[0]}  syntactic head')
            continue
        if len(to_retain_labels) == 0:
            continue
        classes[model] = to_retain_labels
        #print( model, to_retain_labels)
    #print()
    return classes

def createPage(foldername):
    filename = foldername + "/" + "Agreement.html"
    if os.path.exists(filename):
        with open(f"{filename}", 'a') as op:
            op.write(
                f"<li>{args.features}:. <a href=\"{args.features}/{args.features}.html\">Rules</a></li>\n")

    else:
        with open(f"{filename}", 'w') as op:
            HEADER = ORIG_HEADER.replace("main.css", "../../main.css")
            op.write(HEADER + "\n")
            op.write(f'<ul class="nav"><li class="nav"><a class="active" href=\"../../index.html\">Home</a>'
                     f'</li><li class="nav"><a href=\"../../introduction.html\">Usage</a></li>'
                     f'<li class="nav"><a href=\"../../about.html\">About Us</a></li></ul>')
            op.write(f"<br><a href=\"../../index.html\">Back to language list</a><br>")
            op.write(f"<h1> {language_fullname} </h1> <br>\n")
            op.write(
                f'<h3> We  present  a  framework that automatically creates a first-pass specification of  rules for different linguistic phenomena from a raw text corpus for the language in question.</h3>')
            op.write(
                f"<br><strong>{language_fullname}</strong> exhibits the following linguist phenomena for which we extract rules:<br><ul>")
            op.write(
                f"<li>{args.features}:. <a href=\"{args.features}/{args.features}.html\">Rules</a></li>\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data")
    parser.add_argument("--file", type=str, default="./input_files.txt")
    parser.add_argument("--relation_map", type=str, default="./relation_map")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--folder_name", type=str, default='./website/', help="Folder to hold the rules, need to add header.html, footer.html always")

    parser.add_argument("--task", type=str, default="agreement")
    parser.add_argument("--features", type=str, default="Gender+Person+Number")

    parser.add_argument("--sud", action="store_true", default=True, help="Enable to read from SUD treebanks")
    parser.add_argument("--auto", action="store_true", default=False, help="Enable to read from automatically parsed data")
    parser.add_argument("--noise", action="store_true", default=False,
                        help="Enable to read from automatically parsed data")

    parser.add_argument("--prune", action="store_true", default=True)
    parser.add_argument("--binary", action="store_true", default=True)
    parser.add_argument("--retainNA", action="store_true", default=False)

    parser.add_argument("--no_print", action="store_true", default=False,
                        help="To not print the website pages")
    parser.add_argument("--lang", type=str, default=None)

    #Different features to experiment with
    parser.add_argument("--only_triples", action="store_true", default=False, help="Only use relaton, head-pos, dep-pos, disable to use other features")

    parser.add_argument("--use_wikid", action="store_true", default=False, help="Add features from WikiData, requires args.wiki_path and args.wikidata")
    parser.add_argument("--wiki_path", type=str,
                        default="babelnet_outputs/outputs/", help="Contains entity identification for nominals")
    parser.add_argument("--wikidata", type=str,
                        default="./wikidata_processed.txt", help="Contains the WikiData property for each Qfeature")

    parser.add_argument("--lexical", action="store_true", default=True, help="Add lexicalized features for head and dep")

    parser.add_argument("--use_spine", action="store_true", default=False, help="Add features from spine embeddings, read from args.spine_outputs path.")
    parser.add_argument("--spine_outputs", type=str, default="./spine_outputs/")
    parser.add_argument("--continuous", action="store_true", default=True,
                        help="Add features from spine embeddings, read from args.spine_outputs path.")

    parser.add_argument("--best_depth", type=int, nargs='+', default=[-1, -1, -1],
                        help="Set an integer betwen [3,10] to print the website with the required depth,"
                             "order of best-depth ['Gender', 'Person' ,'Number']")
    parser.add_argument("--use_xgboost", action="store_true", default=True)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--n_thread", type=int,default=1)
    parser.add_argument("--skip_models", type=str, nargs='+', default=[])
    parser.add_argument("--transliterate", type=str, default=None)
    args = parser.parse_args()


    folder_name = f'{args.folder_name}'
    with open(f"{folder_name}/header.html") as inp:
        ORIG_HEADER = inp.readlines()
    ORIG_HEADER = ''.join(ORIG_HEADER)

    with open(f"{folder_name}/footer.html") as inp:
        FOOTER = inp.readlines()
    FOOTER = ''.join(FOOTER)

    # populate the best-depth to print the rules for treebank mentioned in args.file
    # If all values are -1 then it will search for the best performing depth by accuracy
    best_depths = {'Gender': args.best_depth[0],
                   'Person': args.best_depth[1],
                   'Number': args.best_depth[2],
                   }

    with open(args.file, "r") as inp:
        files = []
        test_files = defaultdict(set)
        for file in inp.readlines():
            if file.startswith("#"):
                continue
            file_info = file.strip().split()
            if args.lang and args.lang not in file_info:
                continue
            files.append(f'{args.input}/{file_info[0]}')
            if len(file_info) > 1:  # there are test files mentioned
                for test_file in file_info[1:]:
                    test_file = f'{args.input}/{test_file}'
                    test_files[f'{args.input}/{file_info[0]}'].add(test_file.lstrip().rstrip())

    d = {}
    relation_map = {}
    with open(args.relation_map, "r") as inp:
        for line in inp.readlines():
            info = line.strip().split(";")
            key = info[0]
            value = info[1]
            relation_map[key] = (value, info[-1])
            if '@x' in key:
                relation_map[key.split("@x")[0]] = (value, info[-1])

    #Get the wikiparsed data
    features = args.features.split("+")
    for fnum, treebank in enumerate(files):
        train_path, dev_path, test_path, lang = utils.getTreebankPaths(treebank.strip(), args)
        test_files[treebank].add(test_path)
        if train_path is None:
            print(f'Skipping the treebank as no training data!')
            continue

        language_fullname = "_".join(os.path.basename(treebank).split("_")[1:])
        lang_full = lang
        lang_id = lang.split("_")[0]
        if args.auto:
            lang = f'{lang}_auto'
        elif args.noise:
            lang = f'{lang}_noise'

        # train a classifier for each dependent child POS
        try:  # Create the model dir if not already present
            os.mkdir(f"{folder_name}/{lang}/")
        except OSError:
            i = 0

        filename = f"{folder_name}/{lang}/Agreement/"
        try:  # Create the model dir if not already present
            os.mkdir(f"{folder_name}/{lang}/Agreement")
        except OSError:
            i = 0

        # Get spine embedding
        spine_word_vectors, spine_features, spine_dim = None, [], 0
        if args.use_spine:
            spine_word_vectors, spine_features, spine_dim = utils.loadSpine(args.spine_outputs, lang_id,
                                                                             args.continuous)
        genre_train_data, genre_dev_data, genre_test_data, wikiData = utils.getWikiFeatures(args, lang)

        for feature in features:
            if args.skip_models and feature in args.skip_models:
                continue
            print(feature)
            args.features = feature
            try:  # Create the model dir if not already present
                os.mkdir(f"{folder_name}/{lang}/Agreement/{args.features}")
            except OSError:
                i = 0

            with open(f"{folder_name}/{lang}/Agreement/{args.features}/{args.features}.html", 'w') as op:
                HEADER = ORIG_HEADER.replace("main.css", "../../../main.css")
                op.write(HEADER + "\n")
                op.write(f'<ul class="nav"><li class="nav"><a class="active" href=\"../../../index.html\">Home</a>'
                         f'</li><li class="nav"><a href=\"../../../introduction.html\">Usage</a></li>'
                         f'<li class="nav"><a href=\"../../../about.html\">About Us</a></li></ul>')
                op.write(f"<br><a href=\"../../../index.html\">Back to language list</a><br>")
                op.write(f"<h1>  {args.features} agreement for {language_fullname} </h1> <br>\n")



                data_loader = dataloader.DataLoader(args, relation_map)

                # Creating the vocabulary
                input_dir = f"{folder_name}/{lang}/Agreement/"
                vocab_file = input_dir + f'/vocab_{feature}.txt'
                inputFiles = [train_path, dev_path, test_path]
                data_loader.readData(inputFiles, [genre_train_data, genre_dev_data, genre_test_data],
                                     wikiData, args.features,
                                        vocab_file,
                                     spine_word_vectors,
                                     spine_features,
                                     spine_dim
                                     )

                # Get the model info i.e. which models do we have to train e.g. subject-verb, object-verb and so on.
                modelsData = filterData(data_loader)
                if not modelsData:
                    continue

                #createPage(filename)

                for model, to_retain_labels in modelsData.items():
                    if args.skip_models and model in args.skip_models:
                        continue
                    try:  # Create the model dir if not already present
                        os.mkdir(f"{filename}/{args.features}/{model}/")
                    except OSError:
                        i = 0

                    op.write(f"<li>  {args.features} agreement for {model}-words </h1> <a href=\"{model}/{model}.html\"> {model} </a> </li>\n")

                    train_file = f'{filename}/{args.features}/{model}/train.feats.libsvm'  # Stores the features
                    dev_file = f'{filename}/{args.features}/{model}/dev.feats.libsvm'
                    test_file = f'{filename}/{args.features}/{model}/test.feats.libsvm'
                    columns_file = f'{filename}/{args.features}/{model}/column.feats'


                    labels = [0,1]#

                    if not (os.path.exists(train_file) and os.path.exists(test_file)):
                        train_features, columns, output_labels = data_loader.getAssignmentFeatures(train_path, genre_train_data, wikiData, args.features, model, spine_word_vectors, spine_features, spine_dim, train_file)

                        dev_df = None
                        if dev_path:
                            dev_features, _, _ = data_loader.getAssignmentFeatures(dev_path,genre_dev_data,  wikiData, args.features, model, spine_word_vectors, spine_features, spine_dim, dev_file)

                        test_features, _, _ = data_loader.getAssignmentFeatures(test_path, genre_test_data, wikiData, args.features, model, spine_word_vectors, spine_features, spine_dim, test_file)

                    # Reload the df again, as some issue with first loading, print(f'Reading train/dev/test from {lang}/{pos}')
                    print(f'Loading train/dev/test...')
                    label2id, label_list = loadDataLoader(vocab_file, data_loader)

                    train_features, train_labels, \
                    dev_features, dev_labels, \
                    test_features, test_labels, \
                    baseline_label, \
                    id2label, minority, majority = utils.getModelTrainingDataLibsvm(train_file, dev_file, test_file,
                                                                                    label2id,
                                                                                    data_loader.model_data_case[model])

                    y_baseline = [label2id[baseline_label]] * len(test_labels)
                    baseline_acc = accuracy_score(test_labels, y_baseline) * 100.0
                    tree_features = data_loader.feature_map_id2model[model]  # feature_name: feature_id

                    print('Training Started...')
                    train_data_path = train_file.replace("libsvm", "")
                    test_data_path = test_file.replace("libsvm", "")
                    test_datasets = [(test_features, test_labels, lang_full, test_data_path, test_path)]
                    # try:

                    train(train_features, train_labels, train_path, train_data_path,
                          dev_features, dev_labels,
                          test_features, test_labels, test_path, test_data_path,
                          model, baseline_label,
                          f'{filename}/{args.features}/', lang_full, tree_features,
                          relation_map, best_depths,
                          genre_train_data, wikiData, spine_word_vectors, spine_features, spine_dim, test_datasets)
                    # except Exception as e:
                    #except Exception as e:
                    #    print(f'ERROR: Skipping {lang_full} - {model}')
                    #    continue

        with open(f"{folder_name}/{lang}/index_agreement.html", 'a') as outp:
            outp.write("</ul><br><br><br>\n" + FOOTER + "\n")

