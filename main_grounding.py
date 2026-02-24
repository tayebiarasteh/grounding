"""
Created on Feb 23, 2026.
main_grounding.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import pdb
import torch
import os
from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss
from torchvision import transforms, models
import numpy as np
from torch import nn
import pandas as pd
import matplotlib
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoImageProcessor, AutoModel
matplotlib.use('Agg')

from config.serde import open_experiment, create_experiment, delete_experiment, write_config
from Train_Valid_grounding import Training
from Prediction_grounding import Prediction
from data.data_provider import RadiopaediaChunksStore, RetrieverSupervisionDataset, retriever_collate_fn, VerifierSupervisionDataset, verifier_collate_fn
from models.model import BiEncoder, WeightedInfoNCELoss
from Train_Valid_verifier import TrainingVerifier

import warnings
warnings.filterwarnings('ignore')
from huggingface_hub import login




def main_train_retriever(global_config_path="/PATH/config/config.yaml", valid=False, resume=False, experiment_name='name'):
    """Main function for training + validation centrally

        Parameters
        ----------
        global_config_path: str
            always global_config_path="/PATH/config/config.yaml"

        valid: bool
            if we want to do validation

        resume: bool
            if we are resuming training on a model

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]
    # login(token=params["hf_login"])

    base_path = params["paths"]["base_path"]
    chunks_path = os.path.join(base_path, "chunks.parquet")
    chunks_store = RadiopaediaChunksStore(chunks_path)


    train_dataset = RetrieverSupervisionDataset(cfg_path=cfg_path, chunks_store=chunks_store, split_name="train",
        top_k=int(params["retriever"]["top_k_positives"]), min_pos_weight=float(params["retriever"]["min_pos_weight"]))

    valid_dataset = RetrieverSupervisionDataset(cfg_path=cfg_path, chunks_store=chunks_store, split_name="valid",
        top_k=int(params["retriever"]["top_k_positives"]), min_pos_weight=float(params["retriever"]["min_pos_weight"]))


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(params["Network"]["batch_size"]), shuffle=True,  num_workers=10,
        pin_memory=True, collate_fn=lambda b: retriever_collate_fn(b, top_k=int(params["retriever"]["top_k_positives"])), drop_last=True)


    if valid:
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=int(params["Network"]["batch_size"]),
                                                   shuffle=False, num_workers=5, pin_memory=True,
                                                   collate_fn=lambda b: retriever_collate_fn(b, top_k=int(
                                                       params["retriever"]["top_k_positives"])), drop_last=False)
    else:
        valid_loader = None

    # Changeable network parameters
    model = BiEncoder(params["retriever"]["model_name"], tied=bool(params["retriever"]["tied_encoders"]),
                      pooling=params["retriever"]["pooling"])


    loss_function = WeightedInfoNCELoss

    model_info = params['Network']
    model_info['lr'] = float(params["Network"]["lr"])
    model_info['batch_size'] = int(params["Network"]["batch_size"])
    params['Network'] = model_info
    write_config(params, cfg_path, sort_keys=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["Network"]["lr"]),
                                 weight_decay=float(params['Network']['weight_decay']))

    trainer = Training(cfg_path, resume=resume)

    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function)
    else:
        trainer.setup_model(model=model, optimiser=optimizer, loss_function=loss_function)
    trainer.train_epoch(train_loader=train_loader, valid_loader=valid_loader, num_epochs=params['Network']['num_epochs'])





def main_train_verifier(global_config_path="/PATH/config/config.yaml", valid=False, resume=False, experiment_name="name"):
    """
    Main function for verifier (Step 8) training + optional validation.

    Mirrors your retriever main_train_retriever structure.
    """
    if resume is True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)

    cfg_path = params["cfg_path"]

    # datasets
    train_dataset = VerifierSupervisionDataset(cfg_path=cfg_path, split_name="train")
    valid_dataset = VerifierSupervisionDataset(cfg_path=cfg_path, split_name="valid")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(params["Network"]["batch_size"]),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=verifier_collate_fn,
        drop_last=True)

    if valid:
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=int(params["Network"]["batch_size"]),
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=verifier_collate_fn,
            drop_last=False)
    else:
        valid_loader = None

    # model
    model = AutoModelForSequenceClassification.from_pretrained(params["Verifier"]["model_name"],
        num_labels=1,  # regression head
        problem_type="regression")


    # # ---------- memory saver for small GPUs ----------
    # freeze_backbone = bool(params["Verifier"].get("freeze_backbone", False))
    # unfreeze_last_n = int(params["Verifier"].get("unfreeze_last_n_layers", 0))
    #
    # if freeze_backbone:
    #     # Freeze everything first
    #     for p in model.parameters():
    #         p.requires_grad = False
    #
    #     # Unfreeze regression head
    #     for p in model.classifier.parameters():
    #         p.requires_grad = True
    #
    #     # Optionally unfreeze last N encoder layers
    #     if unfreeze_last_n > 0:
    #         enc_layers = model.deberta.encoder.layer
    #         for layer in enc_layers[-unfreeze_last_n:]:
    #             for p in layer.parameters():
    #                 p.requires_grad = True
    #
    #     print(
    #         f"[Verifier] Frozen backbone. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    # # -----------------------------------------------




    # loss (regression)
    loss_function = torch.nn.MSELoss

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=float(params["Network"]["lr"]),
        weight_decay=float(params["Network"]["weight_decay"]))



    # trainable_params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.AdamW(
    #     trainable_params,
    #     lr=float(params["Network"]["lr"]),
    #     weight_decay=float(params["Network"]["weight_decay"]),
    # )





    # write some info into config (same style as retriever)
    model_info = params["Network"]
    model_info["lr"] = float(params["Network"]["lr"])
    model_info["batch_size"] = int(params["Network"]["batch_size"])
    params["Network"] = model_info
    write_config(params, cfg_path, sort_keys=True)

    trainer = TrainingVerifier(cfg_path, resume=resume)

    if resume is True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function)
    else:
        trainer.setup_model(model=model, optimiser=optimizer, loss_function=loss_function)

    trainer.train_epoch(train_loader=train_loader, valid_loader=valid_loader, num_epochs=int(params["Network"]["num_epochs"]))








if __name__ == '__main__':
    delete_experiment(experiment_name="nameverifier", global_config_path=global_config_path)

    # main_train_retriever(global_config_path=global_config_path, valid=True, resume=False, experiment_name='name')

    main_train_verifier(global_config_path=global_config_path, valid=True, resume=False, experiment_name="nameverifier")
