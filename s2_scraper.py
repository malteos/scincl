import json
import logging
import os
import time

import fire
import requests
from tqdm.auto import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_from_ids(input_path, output_dir, id_type='s2', save_every=1000, return_results=False, filter_errors=None):
    """

    Usage: python s2_scraper.py get_from_ids ./data/paperswithcode_arxiv_ids.csv ./data/paperswithcode_s2 --save_every=1000

    :param input_path: Path to ID file (one ID per line)
    :param output_dir: Path to output directory, it will contain id2paper.json and id2error.json
    :param id_type: ID Type (arxiv,acl,mag,...)
    :param save_every: Save output every X query
    :param filter_errors: Comma separated list of error codes (ints) that will be kept (e.g., 404), others will be removed.
    :return: id2paper, id2error
    """

    api_url = 'https://api.semanticscholar.org/v1/paper/'

    # Set ID prefix
    if id_type == 'arxiv':
        api_url += 'arXiv:'
    elif id_type == 'mag':
        api_url += 'MAG:'
    elif id_type == 'acl':
        api_url += 'ACL:'
    elif id_type == 'pubmed':
        api_url += 'PMID:'
    elif id_type == 'corpus':
        api_url += 'CorpusID:'
    elif id_type == 's2':
        # no id prefix for S2 paper ids
        pass

    with open(input_path) as f:
        input_ids = [l.strip() for l in f]

    logger.info(f'Loaded {len(input_ids):,} IDs')

    if not os.path.exists(output_dir):
        logger.info(f'Creating output dir: {output_dir}')
        os.makedirs(output_dir)

    paper_fp = os.path.join(output_dir, 'id2paper.json')
    error_fp = os.path.join(output_dir, 'id2error.json')

    if os.path.exists(paper_fp):
        with open(paper_fp) as f:
            id2paper = json.load(f)
    else:
        id2paper = {}

    if os.path.exists(error_fp):
        with open(error_fp) as f:
            id2error = json.load(f)
        
        # filter errors: keep 404, remove 501 etc..
        if filter_errors is not None and isinstance(filter_errors, str):
            filter_errors_list = filter_errors.split(',')
            filter_errors_list = set([int(error_code) for error_code in filter_errors_list if len(error_code) > 0])
            
            logger.info(f'Errors before filtering: {len(id2error):,}')
            logger.info(f'Keep errors: {filter_errors_list}')
            
            id2error = {i: error_code for i, error_code in id2error.items() if error_code in filter_errors_list}
            
            logger.info(f'Errors after filtering: {len(id2error):,}')
    else:
        id2error = {}

    # Filter for existing ids
    input_ids = [i for i in input_ids if i not in id2paper and i not in id2error]

    logger.info(f'Retrieving {len(input_ids):,} IDs')

    try:
        for i, (input_id) in enumerate(tqdm(input_ids, total=len(input_ids))):
            # Query S2 API
            try:
                res = requests.get(api_url + input_id)

                # Handle response
                if res.status_code == 200:
                    try:
                        id2paper[input_id] = res.json()
                    except ValueError as e:
                        print(f'Error cannot parse JSON: {input_id} - {e}: {res.text}')
                        id2error[input_id] = str(e)
                elif res.status_code == 403:
                    logger.warning(f'Forbidden... probably also some kind of rate limit.. sleep...')
                    time.sleep(60)
                elif res.status_code == 429:
                    logger.warning(f'Stop! Rate limit reached at: {i}')
                    break
                else:
                    logger.error(f'Error status: {res.status_code} - {input_id} - {res.text}')
                    id2error[input_id] = res.status_code

                time.sleep(2.5)  # avoid rate limits

                if i > 0 and (i % save_every) == 0:
                    logger.info(f'Saving at {i:,}...')
                    with open(paper_fp, 'w') as f:
                        json.dump(id2paper, f)

                    with open(error_fp, 'w') as f:
                        json.dump(id2error, f)
            except requests.RequestException as e:
                # probably server error
                logger.info(f'Request error (wait and continue): {e}')
                time.sleep(60)
                
    except KeyboardInterrupt:
        logger.warning('Stopping...')
    finally:
        logger.info(f'Saving...')
        with open(paper_fp, 'w') as f:
            json.dump(id2paper, f)

        with open(error_fp, 'w') as f:
            json.dump(id2error, f)

    logger.info('done')

    if return_results:
        return id2paper, id2error


def convert_json_to_csv_and_jsonl(input_json_path, output_csv_path, output_jsonl_path):
    """

    Usage: python s2_scraper.py convert_json_to_csv_and_jsonl ./data/paperswithcode_s2/id2paper.json ./data/paperswithcode_s2/ids.csv ./data/paperswithcode_s2/papers.jsonl

    :param input_json_path: Key-value JSON dict
    :param output_csv_path: CSV file with one key (ID) per line
    :param output_jsonl_path: JSONL line with one JSON object per line
    :return:
    """

    logger.info(f'Loading from {input_json_path}')

    with open(input_json_path) as f:
        id2data = json.load(f)

    logger.info(f'Loaded {len(id2data):,} key-value pairs')

    logger.info(f'Writing values to {output_jsonl_path}')
    logger.info(f'Writing keys to {output_csv_path}')

    with open(output_csv_path, 'w') as f_csv:
        with open(output_jsonl_path, 'w') as f_jsonl:
            for k, v in tqdm(id2data.items(), total=len(id2data)):
                f_csv.write(k + '\n')
                f_jsonl.write(json.dumps(v) + '\n')

    logger.info('done')


if __name__ == '__main__':
    fire.Fire()
