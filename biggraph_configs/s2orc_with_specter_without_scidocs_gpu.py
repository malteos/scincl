#!/usr/bin/env python3


def get_torchbiggraph_config():
    # Twitter config: https://github.com/facebookresearch/PyTorch-BigGraph/issues/86
    # (Twitter as roughly the same size as S2ORC citation graph) 
    config = {
        # I/O data
        "entity_path": "./data/biggraph/s2orc_with_specter_without_scidocs_gpu",
        "edge_paths": [            
            # "./data/biggraph/s2orc_with_specter_without_scidocs/train_partitioned",
            # copy from CPU version
            # cp -r ./data/biggraph/s2orc_with_specter_without_scidocs/train_partitioned ./data/biggraph/s2orc_with_specter_without_scidocs_gpu
            # cp -r ./data/biggraph/s2orc_with_specter_without_scidocs/entity_names_paper_id_0.json ./data/biggraph/s2orc_with_specter_without_scidocs_gpu/
            # cp -r ./data/biggraph/s2orc_with_specter_without_scidocs/entity_count_paper_id_0.txt ./data/biggraph/s2orc_with_specter_without_scidocs_gpu/
            "./data/biggraph/s2orc_with_specter_without_scidocs_gpu/train_partitioned",

        ],
        "checkpoint_path": "./data/biggraph/s2orc_with_specter_without_scidocs_gpu",
        "checkpoint_preservation_interval": 10,

        # Graph structure
        "entities": {
            "paper_id": {"num_partitions": 4},
        },
        "relations": [{"name": "citation", "lhs": "paper_id", "rhs": "paper_id", "operator": "none"}],

        # Scoring model
        "dimension": 768, #300,  # 500,  # maybe d=768 to make it comparable to BERT
        "max_norm": 1.0,
        "global_emb": False,
        "comparator": "dot",

        # Training
        "num_epochs": 20, #30
        "num_edge_chunks": 10,
        # Since training is being performed on a single GPU instead of 40 cores, the batch size can be increased by about that factor as well.
        "batch_size": 5000,
        # Finally, to take advantage of GPU speed, we suggest turning up num_uniform_negatives and/or num_batch_negatives to about 1000 rather than their default values of 50 (FB15k already uses 1000 uniform negatives).
        "num_uniform_negs": 100,
        "margin": 0.15,
        "lr": 0.1,  # 0.1,

        # GPU
        "num_gpus": 4,  # train partioned with gpu=4
        
        # Evaluation during training
        "eval_fraction": 0,  # to reproduce results, we need to use all training data
    }
    
    return config
