#!/usr/bin/env python3

def get_torchbiggraph_config():

    # Twitter config: https://github.com/facebookresearch/PyTorch-BigGraph/issues/86
    # (Twitter as roughly the same size as S2ORC citation graph) 
    config = {
        # I/O data
        "entity_path": "./data/biggraph/s2orc_train_test",
        "edge_paths": [            
            "./data/biggraph/s2orc_train_test/train_partitioned",
            "./data/biggraph/s2orc_train_test/test_partitioned",
        ],
        "checkpoint_path": "./data/biggraph/models/s2orc_train_test_768d_dot",
        "checkpoint_preservation_interval": 10,

        # Graph structure
        "entities": {
            "paper_id": {"num_partitions": 1},
        },
        "relations": [{"name": "citation", "lhs": "paper_id", "rhs": "paper_id", "operator": "none"}],

        # Scoring model
        "dimension": 768, #300,  # 500,  # maybe d=768 to make it comparable to BERT
        "max_norm": 1.0,
        "global_emb": False,
        "comparator": "dot",

        # Training
        "num_epochs": 30,
        "num_edge_chunks": 10,
        "batch_size": 10_000,
        "num_uniform_negs": 0,
        "margin": 0.15,
        "lr": 0.1,  # 0.1,

        # GPU
        #"num_gpus": 2,
        
        # Evaluation during training
        "eval_fraction": 0,  # to reproduce results, we need to use all training data
    }
    
    return config
