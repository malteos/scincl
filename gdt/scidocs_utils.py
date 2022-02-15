import random
from collections import defaultdict
from typing import Set

import pandas as pd
from tqdm.auto import tqdm


def get_paper_ids_by_dataset(scidocs_dir, ds: str) -> Set[str]:
    tasks = ['recomm', 'coread', 'coview', 'cocite', 'cite', 'mesh', 'mag']
    pids = []

    for task in tasks:

        if task == 'recomm':
            pids += get_paper_ids_from_recomm(scidocs_dir + f'/{task}/{ds}.csv')

        elif task in ['coread', 'coview', 'cocite', 'cite']:
            pids += get_paper_ids_from_qrel(scidocs_dir + f'/{task}/{ds}.qrel')

        elif task in ['mesh', 'mag']:
            pids += get_paper_ids_from_class_csv(scidocs_dir + f'/{task}/{ds}.csv')
        else:
            raise ValueError(f'Invalid task: {task}')

    # unique
    pids = set(pids)

    return pids


def get_paper_ids_from_class_csv(input_path):
    df = pd.read_csv(input_path)

    return list(df.pid.unique())


def get_triples_from_class_csv(input_path):
    df = pd.read_csv(input_path)
    label_to_pids = defaultdict(set)
    pid_to_labels = defaultdict(set)

    for label, group_df in df.groupby('class_label'):
        for pid in group_df.pid.unique():
            label_to_pids[label].add(pid)
            pid_to_labels[pid].add(label)

    pids = list(pid_to_labels.keys())

    triples = []

    for label, label_pids in tqdm(label_to_pids.items(), total=len(label_to_pids)):
        for query in label_pids:
            for pos in label_pids:
                if query != pos:
                    # sample negatives => any pid with different label
                    neg = None
                    while neg is None:
                        candidate = random.choice(pids)
                        candidate_labels = pid_to_labels[candidate]

                        if len(candidate_labels & pid_to_labels[query]) == 0:
                            neg = candidate

                    triples.append((query, pos, neg))

    return triples


def get_paper_ids_from_recomm(input_path):
    recomm_test_df = pd.read_csv(input_path)

    pids = []

    pids += list(recomm_test_df.pid.unique())
    pids += [i for l in recomm_test_df['similar_papers_avail_at_time'].tolist() for i in l.split(',')]

    return pids


def get_recomm_triples(input_path):
    recomm_test_df = pd.read_csv(input_path)
    triples = []

    for query, group_df in recomm_test_df.groupby('pid'):
        clicked_pids = set(group_df['clicked_pid'].tolist())
        viewed_pids = set([i for l in group_df['similar_papers_avail_at_time'].tolist() for i in l.split(',')])
        not_clicked_pids = viewed_pids - clicked_pids

        clicked_pids = list(clicked_pids)
        not_clicked_pids = list(not_clicked_pids)

        for i in range(min(len(clicked_pids), len(not_clicked_pids))):
            triples.append((
                query,
                clicked_pids[i],
                not_clicked_pids[i],
            ))

    return triples


def get_paper_ids_from_qrel(input_path):
    import pytrec_eval

    with open(input_path, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)

    pids = []

    for query, rels in tqdm(qrel.items(), total=len(qrel)):
        pids.append(query)
        for pid, val in rels.items():
            pids.append(pid)

    return pids


def get_triples_from_qrel(input_path):
    import pytrec_eval

    with open(input_path, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)

    triples = []

    for query, rels in tqdm(qrel.items(), total=len(qrel)):
        pos_pids = []
        neg_pids = []
        for pid, val in rels.items():
            if val == 1:
                pos_pids.append(pid)
            else:
                neg_pids.append(pid)

        for i in range(min(len(pos_pids), len(neg_pids))):
            triples.append((
                query,
                pos_pids[i],
                neg_pids[i],
            ))
    return triples
