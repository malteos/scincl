import json
import logging
import os

import fire
import h5py
import numpy as np
from scidocs import get_scidocs_metrics
from scidocs.paths import DataPaths
from smart_open import open

from experiments.environment import get_env
from gdt.utils import flatten, get_scidocs_metadata

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def extract_graph_embeddings(
        run_name: str,
        scidocs_dir: str,
        paper_ids_path: str,
        embeddings_path: str,
        s2id_to_s2orc_paper_id_path: str,
        do_eval: bool = False,
        val_or_test: str = 'test',
        workers: int = 0):
    """
    Extract graph embeddings (from BigGraph) and write into SciDocs format.

    Output files: [mag_mesh/recomm/view_cite_read]__{method}.jsonl

    python cli_scidocs.py extract_graph_embeddings foo \
        --scidocs_dir ${SCIDOCS_DIR} \
        --paper_ids_path ${S2ORC_PAPER_IDS} \
        --embeddings_path ${S2ORC_EMBEDDINGS} \
        --s2id_to_s2orc_paper_id_path ${SPECTER_DIR}/s2id_to_s2orc_paper_id.json

    :param val_or_test: Evaluate on validation or test set (val, test)
    :param workers:
    :param do_eval:
    :param run_name: Run name for output file name
    :param scidocs_dir:
    :param paper_ids_path:
    :param embeddings_path:
    :param s2id_to_s2orc_paper_id_path:
    :return:
    """

    env = get_env()

    if workers < 1:
        workers = env['workers']

    with open(paper_ids_path, "rt") as tf:
        paper_ids = json.load(tf)  # S2ORC Ids

    logger.info(f'Paper IDs: {len(paper_ids):,}')

    # SciDocs Metadata
    scidocs_metadata = get_scidocs_metadata(scidocs_dir)

    # load ID mapping
    with open(s2id_to_s2orc_paper_id_path) as f:
        s2id_to_s2orc_paper_id = json.load(f)

    # reading all embeddings at once is quicker than loading with indices
    with h5py.File(embeddings_path, "r") as hf:
        embeddings = hf["embeddings"][:, :]

    logger.info(f'Graph embeddings: {embeddings.shape}')

    # Find indices in graph embeddings
    s2orc_paper_id_to_paper_idx = {pid: idx for idx, pid in enumerate(paper_ids)}

    # write extracted embeddings to disk (scidocs format)
    for ds, ds_metadata in scidocs_metadata.items():
        out_fp = os.path.join(scidocs_dir, f'{ds}__{run_name}.jsonl')

        logger.info(ds)

        missing_s2orc = []
        missing_embedding = []
        found_paper_ids = 0

        with open(out_fp, 'w') as f:
            for idx, (s2id, paper) in enumerate(ds_metadata.items()):
                # Is in S2ORC and in S2ORC graph embedding
                if s2id in s2id_to_s2orc_paper_id:
                    s2orc_paper_id = s2id_to_s2orc_paper_id[s2id]

                    if s2orc_paper_id in s2orc_paper_id_to_paper_idx:

                        f.write(json.dumps({
                            'paper_id': paper['paper_id'],
                            'title': paper['title'],
                            'embedding': embeddings[s2orc_paper_id_to_paper_idx[s2orc_paper_id], :].tolist(),
                        }) + '\n')
                        found_paper_ids += 1
                    else:
                        missing_embedding.append((s2id, s2orc_paper_id))
                else:
                    missing_s2orc.append(s2id)

        logger.info(f'missing_s2orc = {len(missing_s2orc):,}')
        logger.info(f'missing_embedding = {len(missing_embedding):,}')

        logger.info(f'found = {found_paper_ids:,}')

    del embeddings  # free memory

    if do_eval:
        evaluate(run_name, scidocs_dir, val_or_test, workers)

    logger.info('done')


def evaluate(
        run_name: str,
        scidocs_dir: str,
        val_or_test: str = 'test',
        workers: int = 0,
        embeddings_dir: str = None,
        classification_embeddings_path: str = None,
        user_activity_and_citations_embeddings_path: str = None,
        recomm_embeddings_path: str = None
    ):
    """

    python cli_scidocs.py evaluate biggraph_epoch_${epoch}_300d \
        --scidocs_dir /home/mostendorff/experiments/scidocs/data

    python cli_scidocs.py evaluate specter_1k \
        --scidocs_dir ${SCIDOCS_DIR}
        --recomm_embeddings_path
        --user_activity_and_citations_embeddings_path
        -- classification_embeddings_path

    :param embeddings_dir:
    :param recomm_embeddings_path:
    :param user_activity_and_citations_embeddings_path:
    :param classification_embeddings_path:
    :param run_name:
    :param scidocs_dir:
    :param val_or_test:
    :param workers:
    :return:
    """
    env = get_env()

    if workers < 1:
        workers = env['workers']

    if embeddings_dir is None:
        embeddings_dir = scidocs_dir

    if classification_embeddings_path is None:
        classification_embeddings_path = os.path.join(embeddings_dir, f'mag_mesh__{run_name}.jsonl')

    if user_activity_and_citations_embeddings_path is None:
        user_activity_and_citations_embeddings_path = os.path.join(embeddings_dir, f'view_cite_read__{run_name}.jsonl')

    if recomm_embeddings_path is None:
        recomm_embeddings_path = os.path.join(embeddings_dir, f'recomm__{run_name}.jsonl')

    # point to the data, which should be in scidocs/data by default
    data_paths = DataPaths(scidocs_dir)

    logger.info(f'Starting SciDocs evaluation for {run_name}')

    # now run the evaluation
    eval_metrics = get_scidocs_metrics(
        data_paths,
        classification_embeddings_path,
        user_activity_and_citations_embeddings_path,
        recomm_embeddings_path,
        val_or_test=val_or_test,  # set to 'val' if tuning hyperparams
        n_jobs=workers,  # the classification tasks can be parallelized
        cuda_device=-1,  # the recomm task can use a GPU if this is set to 0, 1, etc
    )

    logger.info(f'Evaluation metrics: {eval_metrics}')

    # save
    eval_dir = os.path.join(scidocs_dir, 'eval_metrics')

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # Compute avg
    eval_metrics['avg'] = np.mean([v for k, v in flatten(eval_metrics).items()])

    with open(os.path.join(eval_dir, run_name + '.' + val_or_test + '.json'), 'w') as f:
        json.dump(eval_metrics, f)

    logger.info(eval_metrics)

    logger.info(eval_metrics['avg'])


def extract_ids(scidocs_dir):
    """

    For each SciDocs dataset saves <ds>.ids file.

    :param scidocs_dir:
    :return:
    """

    # SciDocs Metadata
    scidocs_metadata = get_scidocs_metadata(scidocs_dir)

    for ds, metadata in scidocs_metadata.items():
        ids_str = '\n'.join(list(metadata.keys()))
        with open(scidocs_dir + '/' + ds + '.ids', 'w') as f:
            f.write(ids_str)

    logger.info('done')


if __name__ == '__main__':
    fire.Fire()
