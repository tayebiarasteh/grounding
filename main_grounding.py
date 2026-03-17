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
import yaml
import json

from config.serde import open_experiment, create_experiment, delete_experiment, write_config
from Train_Valid_retriever import Training
from data.data_provider import RetrieverSupervisionDataset, RadiopaediaSentenceStore, retriever_collate_fn, VerifierBinaryDataset, verifier_binary_collate_fn
from models.model import BiEncoder
from Train_Valid_verifier import TrainingVerifier
from data.mimic_data_preprocessing import RadiologyVerifierBinaryDatasetBuilder, MimicGroundingDatasetBuilder
from data.chexpertplus_data_preprocessing import CheXpertRadiopaediaStep4ExternalTestBuilder, create_fixed_chexpert_test_eval_parquet
from Tier1_2_3_prediction import S4Tier123End2EndEvaluatorTestOnly, S2Tier123End2EndEvaluatorTestOnly, S3Tier123End2EndEvaluatorTestOnly, S1Tier123CaseOnlyEvaluatorTestOnly, CrossSystemComparatorPredOnlyTestOnly, analyze_retrieval_error_coupling


import warnings
warnings.filterwarnings('ignore')








def main_train_retriever(global_config_path="/PATH/config/config.yaml", valid=False, resume=False, experiment_name="name"):
    if resume is True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)

    cfg_path = params["cfg_path"]

    base_path = params["Preprocessing"]["base_path_evidenceData"]
    sentences_path = os.path.join(base_path, "evidence_sentences.parquet")

    sentence_store = RadiopaediaSentenceStore(
        evidence_sentences_parquet=sentences_path)

    train_dataset = RetrieverSupervisionDataset(cfg_path=cfg_path, sentence_store=sentence_store, split_name="train")

    valid_dataset = RetrieverSupervisionDataset(cfg_path=cfg_path, sentence_store=sentence_store, split_name="val")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(params["Network"]["batch_size"]),
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=retriever_collate_fn,
        drop_last=True,
        persistent_workers=True,
    )

    if valid:
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=int(params["Network"]["batch_size"]),
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=retriever_collate_fn,
            drop_last=False,
            persistent_workers=True,
        )
    else:
        valid_loader = None

    model = BiEncoder(
        model_name=params["retriever"]["model_name"],
        tied=bool(params["retriever"]["tied_encoders"]),
        pooling=params["retriever"]["pooling"],
    )

    # training loop now uses logits + integer targets
    loss_function = torch.nn.CrossEntropyLoss

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["Network"]["lr"]),
        weight_decay=float(params["Network"]["weight_decay"]),
    )

    model_info = params["Network"]
    model_info["lr"] = float(params["Network"]["lr"])
    model_info["batch_size"] = int(params["Network"]["batch_size"])
    params["Network"] = model_info
    write_config(params, cfg_path, sort_keys=True)

    trainer = Training(cfg_path, resume=resume)

    if resume is True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function)
    else:
        trainer.setup_model(model=model, optimiser=optimizer, loss_function=loss_function)

    trainer.train_epoch(train_loader=train_loader,
        valid_loader=valid_loader, num_epochs=int(params["Network"]["num_epochs"]))




def _build_binary_class_weights_from_dataset(dataset: VerifierBinaryDataset) -> torch.Tensor:
    counts = dataset.get_label_counts()

    n_neg = max(int(counts.get(0, 0)), 1)
    n_pos = max(int(counts.get(1, 0)), 1)

    # class 0 = not_supported, class 1 = supported
    weights = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32)
    return weights


def main_train_verifier_binary(global_config_path="/PATH/config/config.yaml",
    valid: bool = True,
    resume: bool = False,
    experiment_name: str = "verifier_binary"):

    if resume:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)

    cfg_path = params["cfg_path"]

    train_dataset = VerifierBinaryDataset(cfg_path=cfg_path,
        split_name="train",
        max_evidence_sentences=int(params["Verifier"]["max_evidence_sentences"]))

    valid_dataset = VerifierBinaryDataset(cfg_path=cfg_path,
        split_name="val",
        max_evidence_sentences=int(params["Verifier"]["max_evidence_sentences"]))

    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=int(params["Network"]["batch_size"]),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=verifier_binary_collate_fn,
        drop_last=True,
        persistent_workers=True)

    if valid:
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
            batch_size=int(params["Network"]["batch_size"]),
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=verifier_binary_collate_fn,
            drop_last=False,
            persistent_workers=True)
    else:
        valid_loader = None

    model = AutoModelForSequenceClassification.from_pretrained(
        params["Verifier"]["model_name"],
        num_labels=2,
        problem_type="single_label_classification",
    )

    class_weights = _build_binary_class_weights_from_dataset(train_dataset)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["Network"]["lr"]),
        weight_decay=float(params["Network"]["weight_decay"]),
    )

    model_info = params["Network"]
    model_info["lr"] = float(params["Network"]["lr"])
    model_info["batch_size"] = int(params["Network"]["batch_size"])
    params["Network"] = model_info
    write_config(params, cfg_path, sort_keys=True)

    trainer = TrainingVerifier(cfg_path, resume=resume)

    if resume:
        trainer.load_checkpoint(
            model=model,
            optimiser=optimizer,
            loss_function=torch.nn.CrossEntropyLoss,
            class_weights=class_weights)
    else:
        trainer.setup_model(
            model=model,
            optimiser=optimizer,
            loss_function=torch.nn.CrossEntropyLoss,
            class_weights=class_weights,
        )

    trainer.train_epoch(train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=int(params["Network"]["num_epochs"]))





def main_evaluation(global_config_path, experiment_name, model_weights_path, verifier_weights_path,
                    input_test_data="phase1_fixed_test_eval/mimic_test_eval.parquet", dataset='MIMIC'):

    with open(global_config_path, "r") as f:
        params = yaml.safe_load(f)
    base_path_evidenceData = params["Preprocessing"]["base_path_evidenceData"]


    ''' S1 '''
    s1 = S1Tier123CaseOnlyEvaluatorTestOnly(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        verifier_weights_path=verifier_weights_path,
        device="cuda",
        verifier_top_p=5,
        verifier_max_tokens=1024,
        out_dir=os.path.join(base_path_evidenceData, dataset + "_Results/S1"),
        batch_size_verify=4,
        ece_bins=15,
        seed=13,
        input_test_data=input_test_data,
    )
    s1.run()


    ''' S4 '''
    runner = S4Tier123End2EndEvaluatorTestOnly(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        model_weights_path=model_weights_path,
        verifier_weights_path=verifier_weights_path,
        device="cuda",
        top_k=20,
        verifier_top_p=5,
        verifier_max_tokens=1024,
        out_dir=os.path.join(base_path_evidenceData, dataset + "_Results/S4"),
        batch_size_retrieve=256,
        batch_size_verify=4,
        input_test_data=input_test_data,
        batch_size_embed_docs=256,
        ece_bins=15,
    )
    runner.run()

    results, df_out = analyze_retrieval_error_coupling(
        parquet_path=os.path.join(base_path_evidenceData, dataset + "_Results/S4", "s4_test_predictions.parquet"),
        system_prefix="s4",
        top_k=20,
        make_plot=True,
    )



    ''' S2 '''
    s2 = S2Tier123End2EndEvaluatorTestOnly(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        verifier_weights_path=verifier_weights_path,
        device="cuda",
        top_k=20,
        verifier_top_p=5,
        verifier_max_tokens=1024,
        out_dir=os.path.join(base_path_evidenceData, dataset + "_Results/S2"),
        batch_size_embed_chunks=128,
        batch_size_retrieve=256,
        batch_size_verify=4,
        ece_bins=15,
        input_test_data=input_test_data,
        seed=13,
    )
    s2.run()



    ''' S3 '''
    s3 = S3Tier123End2EndEvaluatorTestOnly(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        verifier_weights_path=verifier_weights_path,
        device="cuda",
        top_k=20,
        verifier_top_p=5,
        verifier_max_tokens=1024,
        out_dir=os.path.join(base_path_evidenceData, dataset + "_Results/S3"),
        batch_size_verify=4,
        ece_bins=15,
        seed=13,
        input_test_data=input_test_data,
        bm25_k1=1.2,
        bm25_b=0.75,
    )
    s3.run()


    ''' Cross comparison '''
    base = params["Preprocessing"]["base_path_evidenceData"]

    comparator = CrossSystemComparatorPredOnlyTestOnly(
        system_test_predictions={
            "S1": os.path.join(base, dataset + "_Results/S1", "s1_test_predictions.parquet"),
            "S2": os.path.join(base, dataset + "_Results/S2", "s2_test_predictions.parquet"),
            "S3": os.path.join(base, dataset + "_Results/S3", "s3_test_predictions.parquet"),
            "S4": os.path.join(base, dataset + "_Results/S4", "s4_test_predictions.parquet"),
        },
        system_full_results_csvs={
            "S1": os.path.join(base, dataset + "_Results/S1", "S1_full_results.csv"),
            "S2": os.path.join(base, dataset + "_Results/S2", "S2_full_results.csv"),
            "S3": os.path.join(base, dataset + "_Results/S3", "S3_full_results.csv"),
            "S4": os.path.join(base, dataset + "_Results/S4", "S4_full_results.csv"),
        },
        system_faithfulness_predictions={
            "S2": os.path.join(base, dataset + "_Results/S2", "s2_tier2_faithfulness_predictions.parquet"),
            "S3": os.path.join(base, dataset + "_Results/S3", "s3_tier2_faithfulness_predictions.parquet"),
            "S4": os.path.join(base, dataset + "_Results/S4", "s4_tier2_faithfulness_predictions.parquet"),
        },
        out_path=os.path.join(
            base,
            dataset + "_Results",
            "cross_system_results",
            "s1_s4_cross_system_comparison.json",
        ),
        ece_bins=15,
        n_boot=1000,
        seed=13,
        verbose=True,
        use_unordered_pairs_only=True,
    )

    comparator.run()









