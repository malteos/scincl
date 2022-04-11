# SciNCL: Neighborhood Contrastive Learning for Scientific Document Representations with Citation Embeddings

Supplemental materials for our preprint [Neighborhood Contrastive Learning for Scientific Document Representations with Citation Embeddings (PDF available on ArXiv)](http://arxiv.org/abs/2202.06671).
Trained models and datasets are available as [GitHub release files](https://github.com/malteos/scincl/releases) and on [Huggingface model hub](https://huggingface.co/malteos/scincl).

## Requirements

- Python 3.6 (same as Nvidia/PyTorch Docker images)
- CUDA GPU (for Transformers)
- FAISS-GPU >= 1.7.1 (v1.7.0 leads to poor results)

## Installation

Create a new virtual environment for Python 3.6 with Conda or use our Slurm-ready Docker image (see Dockerfile):

```bash
conda create -n repo python=3.6
conda activate repo
conda install -y nb_conda_kernels
```

Clone repository and install dependencies:

```bash
git clone https://github.com/malteos/scincl.git  repo
cd repo
pip install --find-links https://download.pytorch.org/whl/torch_stable.html -r requirements.txt

# Optional: Install PyTorch with specific CUDA version (by default torch==1.8.1+cu111)
pip install --find-links https://download.pytorch.org/whl/torch_stable.html torch==1.8.1+cu102

# Install Spacy model (needed for SciDocs)
python -m spacy download en_core_web_sm
```

## How to use the pretrained model

```python
from transformers import AutoTokenizer, AutoModel

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('malteos/scincl')
model = AutoModel.from_pretrained('malteos/scincl')

papers = [{'title': 'BERT', 'abstract': 'We introduce a new language representation model called BERT'},
          {'title': 'Attention is all you need', 'abstract': ' The dominant sequence transduction models are based on complex recurrent or convolutional neural networks'}]

# concatenate title and abstract with [SEP] token
title_abs = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in papers]

# preprocess the input
inputs = tokenizer(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512)

# inference
result = model(**inputs)

# take the first token ([CLS] token) in the batch as the embedding
embeddings = result.last_hidden_state[:, 0, :]
```


## SciDocs Results

The uploaded model weights are the ones that yielded the best results on SciDocs (`seed=4`).
In the paper we report the SciDocs results as mean over ten seeds.

| **model**         | **mag-f1** | **mesh-f1** | **co-view-map** | **co-view-ndcg** | **co-read-map** | **co-read-ndcg** | **cite-map** | **cite-ndcg** | **cocite-map** | **cocite-ndcg** | **recomm-ndcg** | **recomm-P@1** | **Avg** |
|-------------------|-----------:|------------:|----------------:|-----------------:|----------------:|-----------------:|-------------:|--------------:|---------------:|----------------:|----------------:|---------------:|--------:|
| Doc2Vec           |       66.2 |        69.2 |            67.8 |             82.9 |            64.9 |             81.6 |         65.3 |          82.2 |           67.1 |            83.4 |            51.7 |           16.9 |    66.6 |
| fasttext-sum      |       78.1 |        84.1 |            76.5 |             87.9 |            75.3 |             87.4 |         74.6 |          88.1 |           77.8 |            89.6 |            52.5 |             18 |    74.1 |
| SGC               |       76.8 |        82.7 |            77.2 |               88 |            75.7 |             87.5 |         91.6 |          96.2 |           84.1 |            92.5 |            52.7 |           18.2 |    76.9 |
| SciBERT           |       79.7 |        80.7 |            50.7 |             73.1 |            47.7 |             71.1 |         48.3 |          71.7 |           49.7 |            72.6 |            52.1 |           17.9 |    59.6 |
| SPECTER           |         82 |        86.4 |            83.6 |             91.5 |            84.5 |             92.4 |         88.3 |          94.9 |           88.1 |            94.8 |            53.9 |             20 |      80 |
| SciNCL (10 seeds) |       81.4 |        88.7 |            85.3 |             92.3 |            87.5 |             93.9 |         93.6 |          97.3 |           91.6 |            96.4 |            53.9 |           19.3 |    81.8 |
| **SciNCL (seed=4)**   |       81.2 |        89.0 |            85.3 |             92.2 |            87.7 |             94.0 |         93.6 |          97.4 |           91.7 |            96.5 |            54.3 |           19.6 |    81.9 |

Additional evaluations are available in the paper.

## Reproduce experiments

### Data preparations

Download
- [S2ORC 20200705v1](https://github.com/allenai/s2orc)
- SPECTER's original training data: [train.pickle](https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/specter/training-data/train.pkl) (see [here](https://github.com/allenai/specter/issues/2))
- [SciDocs evaluation benchmark](https://github.com/allenai/scidocs)
- Replicated SPECTER w/o leakage: [Pretrained weights and paper ID triples](https://huggingface.co/malteos/specter-wol)


Set the following environment variables accordingly:
```bash
export SPECTER_DIR=
export SCIDOCS_DIR=
export S2ORC_METADATA_DIR=
export DATA_DIR=
export S2ORC_EMBEDDINGS=.h5
export S2ORC_PAPER_IDS=entity_names_paper_id_0.json
export OUTPUT_DIR=
export BASE_MODEL=scibert-scivocab-uncased

```

Download SciDocs data from AWS:
```bash
aws s3 sync --no-sign-request s3://ai2-s2-research-public/specter/scidocs/ ${SCIDOCS_DIR}
```


Extract SPECTER's training data:
```bash
python cli_specter.py extract_triples ${SPECTER_DIR}/train.pkl ${SPECTER_DIR}/train_triples.csv
```

Scrape missing training paper data:
```bash
# extract ids
python cli_triples.py extract_ids_from_triples ${SPECTER_DIR}/train_triples.csv ${SPECTER_DIR}/s2 s2_ids.csv query_s2_ids.csv

# query papers
python s2_scraper.py get_from_ids ${SPECTER_DIR}/s2/query_s2_ids.csv ${SPECTER_DIR}/s2 --save_every=1000

# all ids (NOT recommended, would take approx. 250 hrs - we did not do this for the paper)
python s2_scraper.py get_from_ids ${SPECTER_DIR}/s2/s2_ids.csv ${SPECTER_DIR}/s2 --save_every=1000
```

Extract paper IDs from S2ORC

```bash
# Extract PDF hashes from S2ORC
python cli_s2orc.py get_pdf_hashes ${S2ORC_PDF_PARSES_DIR} \
        ${DATASETS_DIR}/s2orc/20200705v1/full/paper_id_to_pdf_hash.json ${S2ORC_PDF_HASH_TO_ID}

# SciDocs-S2ORC mapping with titles
python cli_s2orc.py get_scidocs_title_mapping ${SCIDOCS_DIR} ${S2ORC_METADATA_DIR} ${DATA_DIR}/scidocs_s2id_to_s2orc_paper_id.json

# Merge SciDocs-S2ORC mappings (from S2 API) 
python cli_s2orc.py get_s2orc_scidocs_mappings \
    ${SPECTER_DIR}/id2paper.json,${SPECTER_DIR}/specter_train_source_papers/id2paper.json,${SCIDOCS_DIR}/scidocs_s2orc/id2paper.json \
    ${DATA_DIR}/scidocs_s2id_to_s2orc_paper_id.json \
    ${SPECTER_DIR}/s2id_to_s2orc_paper_id.json

# Save SPECTER S2ORC IDs
python cli_s2orc.py get_s2orc_paper_ids_from_mapping \
        --mapping_path ${SPECTER_DIR}/s2id_to_s2orc_paper_id.latest.json \
        --output_path ${SPECTER_DIR}/train_s2orc_paper_ids.json
        
# Save SciDocs S2ORC IDs
python cli_s2orc.py get_s2orc_paper_ids_from_mapping \
        --mapping_path ${BASE_DIR}/data/scidocs_s2orc/s2id_to_s2orc_paper_id.latest.json \
        --output_path ${BASE_DIR}/data/scidocs_s2orc/s2orc_paper_ids.json
```

Extract citations from S2ORC
```
# Extract all citations
python cli_s2orc.py get_citations ${S2ORC_METADATA_DIR} ${DATA_DIR}/biggraph/s2orc_full

# Extract citations graph edges from S2ORC: train/test ratio = 1% 
python cli_s2orc.py get_citations ${S2ORC_METADATA_DIR} {DATA_DIR}/biggraph/s2orc_train_test --test_ratio 0.01

# Extract citations of SPECTER training data
python cli_s2orc.py get_citations ${S2ORC_METADATA_DIR}  ./data/biggraph/specter_train/ \
    --included_paper_ids data/specter/train_s2orc_paper_ids.json

# Extract citations except SciDocs papers
python cli_s2orc.py get_citations ${S2ORC_METADATA_DIR} ./data/biggraph/s2orc_without_scidocs/ \
    --excluded_paper_ids data/scidocs_s2orc/s2orc_paper_ids.json  
```

### Training corpus

```bash
# Replicated SPECTER data (w/ leakage)
python cli_s2orc.py get_specter_corpus --s2orc_paper_ids ${BASE_DIR}/data/biggraph/s2orc_full/entity_names_paper_id_0.json \
        --s2orc_metadata_dir ${S2ORC_METADATA_DIR} \
        --scidocs_dir ${SCIDOCS_DIR} \
        --specter_triples_path ${SPECTER_DIR}/train_triples.csv \
        --s2id_to_s2orc_input_path ${ID_MAPPINGS}  \
        --citations_output_dir ${BASE_DIR}/data/biggraph/s2orc_with_specter_without_scidocs \
        --paper_ids_output_dir ${BASE_DIR}/data/sci/s2orc_with_specter_without_scidocs
        
# Random S2ORC subset (w/o leakage)
# 1) use `get_citations` and exclude SciDocs (see above)
# 2) generate SPECTER-like triples
python cli_triples.py get_specter_like_triples \
        --citations_input_path ${BASE_DIR}/data/biggraph/s2orc_without_scidocs/citations.tsv \
        --query_papers ${QUERY_PAPER_COUNT} \
        --triples_count ${TRIPLES_COUNT} \
        --output_path ${MY_SPECTER_DIR}/train_triples.csv \
        --triples_per_query 5 --easy_negatives_count 3 --hard_negatives_count 2 --easy_positives_count 5 \
        --seed ${SEED}
     
```

### Citation graph embeddings

#### Download citation embeddings

- Replicated SPECTER (w/ leakage): [Pretrained embeddings (H5; 150 GB)](https://static.openlegaldata.io/scincl/s2orc_with_specter_without_scidocs/embeddings_paper_id_0.v200.h5), [paper IDs (JSON)](https://static.openlegaldata.io/scincl/s2orc_with_specter_without_scidocs/entity_names_paper_id_0.json)
- Random S2ORC (w/o leakage): [Pretrained embeddings (H5; 150 GB)](https://static.openlegaldata.io/scincl/s2orc_without_scidocs/embeddings_paper_id_0.v200.h5), [paper IDs (JSON)](https://static.openlegaldata.io/scincl/s2orc_without_scidocs/entity_names_paper_id_0.json)

#### Train citation embedding model

Select config file:
```bash
# or other config files
export BIGGRAPH_CONFIG=./biggraph_configs/s2orc_768d_dot.py
```

Train and evaluate (adjust paths for full data set):
```bash
# Import TSV (train and test)
# - train Nodes: 52620852 Edges:  462 912 337
# - test Nodes: 52620852 Edges:     4 675 883
torchbiggraph_import_from_tsv --lhs-col=0 --rhs-col=1 ${BIGGRAPH_CONFIG} \
    ./data/biggraph/s2orc_train_test/citations.train.tsv \
    ./data/biggraph/s2orc_train_test/citations.test.tsv


# Train model on train set (takes 6 hrs)
torchbiggraph_train ${BIGGRAPH_CONFIG} \
    -p edge_paths=./data/biggraph/s2orc_train_test/train_partitioned


# Evaluate on test set (takes 3 min)
torchbiggraph_eval ${BIGGRAPH_CONFIG} \
    -p edge_paths=./data/biggraph/s2orc_train_test/test_partitioned

# Train full S2ORC model
export BIGGRAPH_CONFIG=./biggraph_configs/s2orc_full.py
torchbiggraph_train ${BIGGRAPH_CONFIG} \
    -p edge_paths=./data/biggraph/s2orc_full/train_partitioned

# Train `s2orc_with_specter_without_scidocs` model
export BIGGRAPH_CONFIG=./biggraph_configs/s2orc_with_specter_without_scidocs.py
taskset -c 10-59 torchbiggraph_train ${BIGGRAPH_CONFIG} \
    -p edge_paths=./data/biggraph/s2orc_with_specter_without_scidocs/train_partitioned

```

### kNN search index

To retrieve positve and negative samples from the citation neighborhood, 
we use either k nearest neighbor search or cosine similarity threshold search.
Both search queries are implemented with the [FAISS framework](https://github.com/facebookresearch/faiss)
 (kNN can be also done with [Annoy](https://github.com/spotify/annoy) but yields worse results):

```bash
# Select GPU with at least 24 GB (other decrease batch size)
export CUDA_VISIBLE_DEVICES=

# See https://github.com/facebookresearch/faiss/wiki/The-index-factory
export ANN_FAISS_FACTORY=Flat
export ANN_INDEX_PATH=${EXP_DIR}/{$ANN_FAISS_FACTORY}.faiss

# Build ANN index only for papers in training corpus
python cli_graph.py build_faiss ${S2ORC_EMBEDDINGS} \
    ${ANN_INDEX_PATH} \
    --string_factory ${ANN_FAISS_FACTORY} \
    --paper_ids ${S2ORC_PAPER_IDS} --include_paper_ids ${EXP_DIR}/s2orc_paper_ids.json \
    --do_normalize \
    --batch_size 512 --workers ${WORKERS} --device 0
```


### Contrastive language model

Our full pipeline can be run within a standard Python environment or as Slurm job. Set `PY` variable following accordingly:
```
# standard python
export PY="python"

# slurm (adjust with your settings)
export PY="srun ... python"
```

Run full pipeline including query paper selection, ANN index creation, triple mining, metadata extraction, training, and evaluation:

```bash
${PY} cli_pipeline.py run_specter ${OUTPUT_DIR} \
    --auto_output_dir \
    --scidocs_dir ${SCIDOCS_DIR} \
    --s2orc_metadata_dir ${S2ORC_METADATA_DIR} \
    --specter_triples_path ${SPECTER_DIR}/train_triples.csv \
    --graph_paper_ids_path ${S2ORC_PAPER_IDS} \
    --graph_embeddings_path ${S2ORC_EMBEDDINGS}  \
    --s2id_to_s2orc_input_path ${SPECTER_DIR}/s2id_to_s2orc_paper_id.json \
    --graph_limit specter \
    --ann_metric inner_product --ann_workers 1 \
    --ann_backend faiss \
    --ann_index_path ${ANN_INDEX_PATH} \
    --val_or_test_or_both both --eval_steps 1 --save_steps 2 \
    --triples_per_query 5 \
    --workers ${WORKERS} --gzip \
    --base_model_name_or_path ${BASE_MODEL} \
    --easy_positives_count 5 --easy_positives_strategy knn --easy_positives_k_min 20 --easy_positives_k_max 25 \
    --easy_negatives_count 3 --easy_negatives_strategy random_without_knn \
    --hard_negatives_count 2 --hard_negatives_strategy knn  --hard_negatives_k_min 3998 --hard_negatives_k_max 4000

```

The exact scripts to reproduce our experiments are in the `/sbin` directory.
Evaluation results are reported to [Weights & Biases](https://wandb.com) or stored on disk.

The individual pipeline steps can be run separately or reused to save compute time by setting corresponding arguments (e.g., `--skip-triples`).

## How to cite

If you are using our code or data, please cite [our paper](https://arxiv.org/abs/2202.06671):

```bibtex
@article{Ostendorff2022scincl,
  title={Neighborhood Contrastive Learning for Scientific Document Representations with Citation Embeddings},
  author={Ostendorff, Malte and Rethmeier, Nils and Augenstein, Isabelle and Gipp, Bela and Rehm, Georg},
  journal={arXiv preprint arXiv:2202.06671},
  year={2022}
}
```

## License

MIT
