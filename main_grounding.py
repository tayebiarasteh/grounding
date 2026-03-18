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
from data.data_provider import RetrieverSupervisionDataset, RadiopaediaSentenceStore, retriever_collate_fn, VerifierBinaryDataset, verifier_binary_collate_fn, VerifierBinaryTestDataset
from models.model import BiEncoder
from Train_Valid_verifier import TrainingVerifier
from data.mimic_data_preprocessing import RadiologyVerifierBinaryDatasetBuilder, MimicGroundingDatasetBuilder
from data.chexpertplus_data_preprocessing import CheXpertRadiopaediaStep4ExternalTestBuilder, create_fixed_chexpert_test_eval_parquet
from Tier1_2_3_prediction import S4Tier123End2EndEvaluatorTestOnly, S2Tier123End2EndEvaluatorTestOnly, S3Tier123End2EndEvaluatorTestOnly, S1Tier123CaseOnlyEvaluatorTestOnly, CrossSystemComparatorPredOnlyTestOnly, analyze_retrieval_error_coupling
from Prediction import PredictionVerifier


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





def main_test_verifier_bootstrap(
    global_config_path="/PATH/config/config.yaml",
    experiment_name="verifier_exp",
    experiment_epoch_num=3,
    scenario_name="oracle_indomain",
    test_parquet_path="/PATH/to/test_scenario.parquet",
    split_name="test",
    n_bootstrap=1000,
):
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]

    test_dataset = VerifierBinaryTestDataset(
        cfg_path=cfg_path,
        test_parquet_path=test_parquet_path,
        split_name=split_name,
        max_evidence_sentences=2,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=int(params["Network"]["batch_size"]),
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=2,
        collate_fn=verifier_binary_collate_fn,
        persistent_workers=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        params["Verifier"]["model_name"],
        num_labels=2,
        problem_type="single_label_classification",
    )

    predictor = PredictionVerifier(cfg_path=cfg_path, split_name=split_name, scenario_name=scenario_name)
    predictor.setup_model(model=model, epoch_num=experiment_epoch_num)

    pred_dict = predictor.predict_only(test_loader)

    metrics_dict = predictor._compute_metrics(
        probs=pred_dict["probs"],
        targets=pred_dict["targets"],
    )

    predictor.save_predictions_csv(
        pred_dict=pred_dict,
        metrics_dict=metrics_dict,
        scenario_name=scenario_name,
    )

    boot_dict = predictor.bootstrap_metrics(
        probs=pred_dict["probs"],
        targets=pred_dict["targets"],
        n_bootstrap=n_bootstrap,
        seed=13,
    )

    predictor.save_bootstrap_csv(
        boot_dict=boot_dict,
        scenario_name=scenario_name,
    )

    predictor.print_and_save_stats(
        metrics_dict=metrics_dict,
        boot_dict=boot_dict,
        scenario_name=scenario_name,
    )

    return metrics_dict, boot_dict






