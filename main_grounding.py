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
from data.data_provider import RetrieverSupervisionDataset, RadiopaediaSentenceStore, retriever_collate_fn, VerifierBinaryDataset, verifier_binary_collate_fn, VerifierBinaryTestDataset, verifier_binary_vignette_collate_fn, VerifierBinaryVignetteTestDataset
from models.model import BiEncoder
from Train_Valid_verifier import TrainingVerifier
from data.mimic_data_preprocessing import RadiologyVerifierBinaryDatasetBuilder, MimicGroundingDatasetBuilder
from data.chexpertplus_data_preprocessing import CheXpertPlusGroundingDatasetBuilder, CheXpertPlusVerifierBinaryDatasetBuilder
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



def main_train_verifier_binary(global_config_path="/PATH/config/config.yaml", valid=True, resume=False, experiment_name="verifier_binary"):

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

    model = AutoModelForSequenceClassification.from_pretrained(params["Verifier"]["model_name"], num_labels=2,
                                                               problem_type="single_label_classification")

    class_weights = _build_binary_class_weights_from_dataset(train_dataset)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["Network"]["lr"]),
                                  weight_decay=float(params["Network"]["weight_decay"]))

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





def main_test_verifier_bootstrap(global_config_path="/PATH/config/config.yaml", experiment_name="verifier_exp",
    experiment_epoch_num=3, scenario_name="oracle_indomain", test_parquet_path="/PATH/to/test_scenario.parquet", split_name="test", n_bootstrap=1000):

    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]

    test_dataset = VerifierBinaryTestDataset(cfg_path=cfg_path, test_parquet_path=test_parquet_path,
                                             split_name=split_name, max_evidence_sentences=2)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=int(params["Network"]["batch_size"]),
                                              pin_memory=True, drop_last=False, shuffle=False, num_workers=2,
                                              collate_fn=verifier_binary_collate_fn, persistent_workers=True)

    model = AutoModelForSequenceClassification.from_pretrained(params["Verifier"]["model_name"], num_labels=2,
                                                               problem_type="single_label_classification")

    predictor = PredictionVerifier(cfg_path=cfg_path, split_name=split_name, scenario_name=scenario_name)
    predictor.setup_model(model=model, epoch_num=experiment_epoch_num)

    pred_dict = predictor.predict_only(test_loader)

    metrics_dict = predictor._compute_metrics(probs=pred_dict["probs"], targets=pred_dict["targets"])

    predictor.save_predictions_csv(pred_dict=pred_dict, metrics_dict=metrics_dict, scenario_name=scenario_name)

    boot_dict = predictor.bootstrap_metrics(probs=pred_dict["probs"], targets=pred_dict["targets"], n_bootstrap=n_bootstrap, seed=13)

    predictor.save_bootstrap_csv(boot_dict=boot_dict, scenario_name=scenario_name)

    predictor.print_and_save_stats(metrics_dict=metrics_dict, boot_dict=boot_dict, scenario_name=scenario_name)


    return metrics_dict, boot_dict





def main_test_vignette_mcq( global_config_path="/PATH/config/config.yaml", experiment_name="verifier_exp",
                            experiment_epoch_num=3, scenario_name="vignette_mcq",
                            test_parquet_path="/PATH/to/vignette_test.parquet", split_name="test", n_bootstrap=1000):
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]

    test_dataset = VerifierBinaryVignetteTestDataset(cfg_path=cfg_path, test_parquet_path=test_parquet_path, split_name=split_name,
                                          max_evidence_sentences=2)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=int(params["Network"]["batch_size"]),
                                              pin_memory=True, drop_last=False, shuffle=False, num_workers=2,
                                              collate_fn=verifier_binary_vignette_collate_fn, persistent_workers=True)

    model = AutoModelForSequenceClassification.from_pretrained(params["Verifier"]["model_name"], num_labels=2,
                                                               problem_type="single_label_classification")

    predictor = PredictionVerifier(cfg_path=cfg_path, split_name=split_name, scenario_name=scenario_name)

    predictor.setup_model(model=model, epoch_num=experiment_epoch_num)

    pred_dict = predictor.predict_only_mcq(test_loader)

    metrics_dict, boot_dict, question_df = predictor.save_and_report_mcq_stats(pred_dict=pred_dict, scenario_name=scenario_name,
                                                                               n_bootstrap=n_bootstrap, seed=13)

    return metrics_dict, boot_dict, question_df








    # print(df.iloc[2])
    # working_df = df.iloc[2000]
    # print('\nchunk_text:', working_df['chunk_text'])
    # df['soft_support_score'].mean()
    # df['s1_pred_support_raw'].mean()
    # df['s1_pred_prob_sigmoid'].mean()

    # pos = df[df["label_id"] == 1]
    # neg = df[df["label_id"] == 0]
    #
    # neg_sampled = neg.sample(len(pos), random_state=13)
    #
    # df = pd.concat([pos, neg_sampled]).sample(frac=1, random_state=13).reset_index(drop=True)


    def has_concept(row):
        concept = row["concept"].lower()
        sents = json.loads(row["evidence_sentences_json"])
        return any(concept in s.lower() for s in sents)


    c = df[df["row_type"] == "C"]
    print("C explicit concept ratio:", c.apply(has_concept, axis=1).mean())


    def case_mentions_concept(row):
        concept = row["concept"].lower()
        return concept in row["case_context"].lower()


    print("C shortcut leakage:", c.apply(case_mentions_concept, axis=1).mean())

    print("evidence diversity:", df["evidence_sentences_json"].nunique() / len(df))

    print(df["label"].value_counts())
    print(df["row_type"].value_counts())


    # working_df = df.iloc[2000]
    # print('\nqa_row_id:', working_df['qa_row_id'])
    # print('\nsubject_id:', working_df['subject_id'])
    # print('\nquestion:', working_df['question'])
    # print('\nclaim_text:', working_df['claim_text'])
    # print('\ncase_context:', working_df['case_context'])
    # print('\nanswer_target:', working_df['answer_target'])
    # print('\nsoft_support_score:', working_df['soft_support_score'])
    # print('\nevidence_weights_json:', working_df['evidence_weights_json'])
    # print('\nevidence_chunk_ids_json:', working_df['evidence_chunk_ids_json'])
    # print('\nqa_type:', working_df['qa_type'])


    # pdb.set_trace()

    asdfsdf = 2

