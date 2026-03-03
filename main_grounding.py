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

from config.serde import open_experiment, create_experiment, delete_experiment, write_config
from Train_Valid_grounding import Training
from data.data_provider import NonRepeatingChunkSampler, RadiopaediaChunksStore, RetrieverSupervisionDataset, retriever_collate_fn, VerifierSupervisionDataset, verifier_collate_fn
from models.model import BiEncoder, WeightedInfoNCELoss
from Train_Valid_verifier import TrainingVerifier
from data.post_retrievertraining import RadiopaediaStep6ReembedAndFaissBuilder, RadiopaediaRStarWeakEvidenceAligner, RadiopaediaStep6CSupervisionBuilder, RadiopaediaStep7VerifierMaterializer
from data.mimic_data_preprocessing import create_fixed_mimic_test_eval_parquet, RadiopaediaSnapshotBuilder, RadiopaediaWeakEvidenceAligner, RadiopaediaStep4CSupervisionBuilder
from data.chexpertplus_data_preprocessing import ChexpertPlusCaseContextBuilder, ChexpertPlusFactsAndQABuilder, ChexpertPlusRStarWeakEvidenceAligner, ChexpertPlusStep6CSupervisionBuilder, ChexpertPlusStep7VerifierMaterializer, create_fixed_chexpertplus_test_eval_parquet
from Tier1_2_3_prediction import CrossSystemComparatorPredOnly, S1Tier123CaseOnlyEvaluator, S3Tier123End2EndEvaluator, S2Tier123End2EndEvaluator, S4Tier123End2EndEvaluator
from external_validation import S4Tier123End2EndEvaluatorChexpertPlus, S2Tier123End2EndEvaluatorChexpertPlus, S3Tier123End2EndEvaluatorChexpertPlus, S1Tier123CaseOnlyEvaluatorChexpertPlus, CrossSystemComparatorPredOnlyChexpertPlus
from vignette_external_validation import VignetteS4Tier123End2EndEvaluator, VignetteS3Tier123End2EndEvaluator, VignetteS2Tier123End2EndEvaluator, VignetteS1Tier123End2EndEvaluator, CrossSystemComparatorPredOnlyVignette, VignetteEvalPaths
from data.Vignette_processing import RadiologyVignetteProcessor

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

    sampler = NonRepeatingChunkSampler(
        train_dataset,
        chunk_size=200_000,
        seed=13,
        drop_last=True,  # keeps batch count stable
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,  # MUST be False when using sampler
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=verifier_collate_fn,
        drop_last=True,
        persistent_workers=True)



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
    # freeze_backbone = True
    # unfreeze_last_n = 0
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





def end2end_mimic_processing_4_7(global_config_path, experiment_name, model_weights_path):

    with open(global_config_path, 'r') as yaml_file:
        params = yaml.safe_load(yaml_file)
    base_path_mimicData = params['Preprocessing']['base_path_mimicData']
    base_path_evidenceData = params['Preprocessing']['base_path_evidenceData']

    ''' step 4a: RadiopaediaSnapshot_v1 (chunking only, no embeddings) '''
    builder = RadiopaediaSnapshotBuilder(
        in_dir=os.path.join(base_path_evidenceData, "articles_clean_txt"),  # folder with ~17k *.txt
        out_dir=base_path_evidenceData,
        snapshot_name=params['Preprocessing']['snapshot_name'],
        tokenizer_name=params['Preprocessing']['tokenizer_name'],
        target_tokens=200,
        min_tokens=50,
        write_csv=False,  # parquet is preferred
    )
    chunks_df = builder.run()
    print(chunks_df.head(10))
    print("n_chunks:", len(chunks_df))
    print("n_articles:", chunks_df["article_id"].nunique())
    print(chunks_df[["chunk_id", "article_id", "article_title", "chunk_token_count"]].head(10))
    #
    # ''' step 4b '''
    builder = RadiopaediaWeakEvidenceAligner(
        qa_csv=os.path.join(base_path_mimicData, "mimic_qa_step3.csv"),
        chunks_parquet=os.path.join(base_path_evidenceData, "chunks.parquet"),
        out_parquet=os.path.join(base_path_evidenceData, "weak_alignment_step4b.parquet"),
        model_name=params['Preprocessing']['tokenizer_name'],
        top_k=20,
        alpha=0.85,
        temperature=0.07,  # keep >0; deterministic with fixed seeds + eval()
        seed=13,
        cache_dir=os.path.join(base_path_evidenceData, "cache_step4b"),
        device="cuda",
        # device="cpu",
        force_cpu_faiss=True,
    )
    builder.run()

    ''' step 4c '''
    builder = RadiopaediaStep4CSupervisionBuilder(
        qa_csv=os.path.join(base_path_mimicData, "mimic_qa_step3.csv"),
        weak_alignment_parquet=os.path.join(base_path_evidenceData, "weak_alignment_step4b.parquet"),
        out_dir=os.path.join(base_path_evidenceData, "step4c_supervision"),
        top_k=20,
        min_weight=0.0,
        qa_chunksize_rows=50_000,
        parquet_batch_rows=50_000,
        subject_id_col="subject_id",  # must exist in mimic_qa_step3.csv
        split_seed=13,
        frac_train=0.75,
        frac_valid=0.10,
        frac_test=0.15,
    )
    builder.run()

    ''' step 6A '''
    builder = RadiopaediaStep6ReembedAndFaissBuilder(
        cfg_path=global_config_path,
        chunks_parquet=os.path.join(base_path_evidenceData, "chunks.parquet"),
        model_weights_path=model_weights_path,
        out_dir=os.path.join(base_path_evidenceData, "step6_faiss_rstar"),
        batch_size=128,
        experiment_name=experiment_name,
        use_amp=True,
        device="cuda",
        seed=13,
    )
    builder.run()

    ''' step 6B '''
    builder = RadiopaediaRStarWeakEvidenceAligner(
        qa_csv=os.path.join(base_path_mimicData, "mimic_qa_step3.csv"),
        chunks_parquet=os.path.join(base_path_evidenceData, "chunks.parquet"),
        faiss_index=os.path.join(base_path_evidenceData, "step6_faiss_rstar/faiss_Rstar.index"),
        out_parquet=os.path.join(base_path_evidenceData, "weak_alignment_step6b_Rstar.parquet"),
        model_name=params['Preprocessing']['tokenizer_name'],
        rstar_query_encoder_ckpt=model_weights_path,
        top_k=20,
        alpha=0.85,
        temperature=0.07,
        seed=13,
        device="cuda",
        use_lexical=True,
    )
    builder.run()

    ''' step 6C '''
    builder = RadiopaediaStep6CSupervisionBuilder(
        qa_csv=os.path.join(base_path_mimicData, "mimic_qa_step3.csv"),
        weak_alignment_parquet=os.path.join(base_path_evidenceData, "weak_alignment_step6b_Rstar.parquet"),
        patient_split_map_json=os.path.join(base_path_evidenceData, "step4c_supervision/patient_split_map.json"),
        out_dir=os.path.join(base_path_evidenceData, "step6c_supervision_rstar"),
        top_k=20,
        min_weight=0.0,
        qa_chunksize_rows=50_000,
        parquet_batch_rows=50_000,
        # allowed_splits=["train", "valid"],
    )
    builder.run()

    ''' step 7 '''
    builder = RadiopaediaStep7VerifierMaterializer(
        verifier_supervision_parquet=os.path.join(
            base_path_evidenceData, "step6c_supervision_rstar/verifier_supervision.parquet"
        ),
        chunks_parquet=os.path.join(base_path_evidenceData, "chunks.parquet"),
        out_dir=os.path.join(base_path_evidenceData, "step7_verifier_materialized_rstar"),
        max_evidence_chunks=20,  # keep consistent with top_k
        min_evidence_weight=0.0,
        parquet_batch_rows=50_000,
        parquet_compression="zstd",
    )
    builder.run()


    ''' Creating the final mimic test parquet '''
    verifier_materialized_parquet = os.path.join(base_path_evidenceData, "step7_verifier_materialized_rstar", "verifier_train_materialized.parquet")
    out_test_eval_parquet = os.path.join(base_path_evidenceData, "phase1_fixed_test_eval", "mimic_test_eval.parquet")
    create_fixed_mimic_test_eval_parquet(verifier_materialized_parquet=verifier_materialized_parquet, out_parquet=out_test_eval_parquet,
        split_value="test", parquet_batch_rows=50_000, parquet_compression="zstd", seed=13, write_manifest_json=True)


    ''' Creating the final mimic valid parquet '''
    verifier_materialized_parquet = os.path.join(base_path_evidenceData, "step7_verifier_materialized_rstar", "verifier_train_materialized.parquet")
    out_valid_eval_parquet = os.path.join(base_path_evidenceData, "phase1_fixed_valid_eval", "mimic_valid_eval.parquet")
    create_fixed_mimic_test_eval_parquet(verifier_materialized_parquet=verifier_materialized_parquet, out_parquet=out_valid_eval_parquet,
        split_value="valid", parquet_batch_rows=50_000, parquet_compression="zstd", seed=13, write_manifest_json=True)




def end2end_chexpert_processing_4_7(global_config_path, experiment_name, model_weights_path):
    with open(global_config_path, 'r') as yaml_file:
        params = yaml.safe_load(yaml_file)
    base_path_evidenceData = params['Preprocessing']['base_path_evidenceData']
    base_path_chexpertData = params["Preprocessing"]["base_path_mimicData"]

    ''' step 6B '''
    builder = ChexpertPlusRStarWeakEvidenceAligner(
        qa_csv=os.path.join(base_path_chexpertData, "chexpertplus_qa_step3.csv"),
        chunks_parquet=os.path.join(base_path_evidenceData, "chunks.parquet"),
        faiss_index=os.path.join(base_path_evidenceData, "step6_faiss_rstar", "faiss_Rstar.index"),
        chunk_ids_npy=os.path.join(base_path_evidenceData, "step6_faiss_rstar", "chunk_ids.npy"),
        out_parquet=os.path.join(base_path_evidenceData, "weak_alignment_step6b_Rstar_chexpertplus.parquet"),
        model_name=params["Preprocessing"]["tokenizer_name"],
        rstar_query_encoder_ckpt=model_weights_path,
        top_k=20,
        alpha=0.85,
        temperature=0.07,
        seed=13,
        device="cuda",
        use_lexical=True,
        # split
        split_seed=13,
        frac_train=0.75,
        frac_valid=0.10,
        frac_test=0.15)
    builder.run()

    ''' step 6C '''
    builder = ChexpertPlusStep6CSupervisionBuilder(
        qa_csv=os.path.join(base_path_chexpertData, "chexpertplus_qa_step3.csv"),
        weak_alignment_parquet=os.path.join(base_path_evidenceData, "weak_alignment_step6b_Rstar_chexpertplus.parquet"),
        out_dir=os.path.join(base_path_evidenceData, "step6c_supervision_rstar_chexpertplus"),
        top_k=20,
        min_weight=0.0,
        qa_chunksize_rows=50_000,
        parquet_batch_rows=50_000)
    builder.run()

    ''' step 7 '''
    builder = ChexpertPlusStep7VerifierMaterializer(
        verifier_supervision_parquet=os.path.join(
            base_path_evidenceData,
            "step6c_supervision_rstar_chexpertplus/verifier_supervision.parquet",
        ),
        chunks_parquet=os.path.join(base_path_evidenceData, "chunks.parquet"),
        out_dir=os.path.join(base_path_evidenceData, "step7_verifier_materialized_rstar_chexpertplus"),
        max_evidence_chunks=20,
        min_evidence_weight=0.0,
        parquet_batch_rows=50_000,
        parquet_compression="zstd")
    builder.run()

    ''' Creating the final mimic test parquet '''
    in_parquet = os.path.join(base_path_evidenceData, "step7_verifier_materialized_rstar_chexpertplus",
        "verifier_train_materialized.parquet")
    out_parquet = os.path.join(base_path_evidenceData, "phase1_fixed_test_eval_chexpertplus",
        "chexpertplus_test_eval.parquet")
    create_fixed_chexpertplus_test_eval_parquet(verifier_materialized_parquet=in_parquet,
        out_parquet=out_parquet, split_value="test", parquet_batch_rows=50_000,
        parquet_compression="zstd", seed=13, write_manifest_json=True)




def main_mimic_evaluation(global_config_path, experiment_name, model_weights_path, verifier_weights_path):
    with open(global_config_path, "r") as f:
        params = yaml.safe_load(f)
    base_path_evidenceData = params["Preprocessing"]["base_path_evidenceData"]


    ''' S1 '''
    s1 = S1Tier123CaseOnlyEvaluator(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        verifier_weights_path=verifier_weights_path,
        device="cuda",
        verifier_max_tokens=int(params["Verifier"]["max_tokens"]),
        out_dir=os.path.join(base_path_evidenceData, "tier123_results_s1_end2end"),
        batch_size_verify=4,
        ece_bins=15,
        seed=13,
        # optional:
        # calib_max_steps=500,
        # calib_lr=0.05,
    )
    s1.run()



    ''' S3 '''
    s3 = S3Tier123End2EndEvaluator(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        verifier_weights_path=verifier_weights_path,
        device="cuda",
        top_k=20,
        verifier_max_tokens=int(params["Verifier"]["max_tokens"]),
        out_dir=os.path.join(base_path_evidenceData, "tier123_results_s3_end2end"),
        batch_size_verify=4,
        ece_bins=15,
        seed=13,
        bm25_k1=1.2,
        bm25_b=0.75,
        # optional speed knobs:
        # faithfulness_max_eval=2000,
        # faithfulness_random_trials=1,
        # calib_max_steps=500,
        # calib_lr=0.05,
    )
    s3.run()




    ''' S2 '''
    s2 = S2Tier123End2EndEvaluator(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        verifier_weights_path=verifier_weights_path,
        device="cuda",
        top_k=20,
        verifier_max_tokens=int(params["Verifier"]["max_tokens"]),
        out_dir=os.path.join(base_path_evidenceData, "tier123_results_s2_end2end"),
        batch_size_embed_chunks=128,
        batch_size_retrieve=256,
        batch_size_verify=4,
        ece_bins=15,
        seed=13,
        # optional speed knobs
        # faithfulness_max_eval=2000,
        # faithfulness_random_trials=1,
        # calib_max_steps=500,
        # calib_lr=0.05,
    )
    s2.run()



    ''' S4 '''
    runner = S4Tier123End2EndEvaluator(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        model_weights_path=model_weights_path,
        verifier_weights_path=verifier_weights_path,
        device="cuda",
        top_k=20,
        verifier_max_tokens=int(params["Verifier"]["max_tokens"]),
        out_dir=os.path.join(base_path_evidenceData, "tier123_results_s4_end2end"),
        batch_size_retrieve=256,
        batch_size_verify=4,
        ece_bins=15,
        seed=13,
        # Tier2 knobs
        # faithfulness_max_eval=2000,
        # faithfulness_random_trials=1,
        # Tier3 knobs
        calibration_method="temperature",
        # calibration_max_eval=10000,
        calibration_opt_steps=200,
        calibration_lr=0.05,
    )
    runner.run()



    ''' Cross comparison '''
    base = params["Preprocessing"]["base_path_evidenceData"]

    comparator = CrossSystemComparatorPredOnly(
        system_test_predictions={
            "S1": os.path.join(base, "tier123_results_s1_end2end", "s1_test_predictions.parquet"),
            "S2": os.path.join(base, "tier123_results_s2_end2end", "s2_test_predictions.parquet"),
            "S3": os.path.join(base, "tier123_results_s3_end2end", "s3_test_predictions.parquet"),
            "S4": os.path.join(base, "tier123_results_s4_end2end", "s4_test_predictions.parquet"),
        },
        system_faithfulness_predictions={
            "S2": os.path.join(base, "tier123_results_s2_end2end", "s2_tier2_faithfulness_predictions.parquet"),
            "S3": os.path.join(base, "tier123_results_s3_end2end", "s3_tier2_faithfulness_predictions.parquet"),
            "S4": os.path.join(base, "tier123_results_s4_end2end", "s4_tier2_faithfulness_predictions.parquet"),
        },
        out_path=os.path.join(base, "cross_system_results", "s1_s4_cross_system_comparison_predonly.json"),
        ece_bins=15,
        n_boot=2000,
        seed=13,
    )
    comparator.run()




def main_chexpert_evaluation(global_config_path, experiment_name, model_weights_path, verifier_weights_path):
    with open(global_config_path, "r") as f:
        params = yaml.safe_load(f)
    base_path_evidenceData = params["Preprocessing"]["base_path_evidenceData"]

    ''' S1 Chexpert'''
    s1 = S1Tier123CaseOnlyEvaluatorChexpertPlus(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        verifier_weights_path=verifier_weights_path,
        device="cuda",
        verifier_max_tokens=int(params["Verifier"]["max_tokens"]),
        out_dir=os.path.join(base_path_evidenceData, "external_validation", "chexpertplus", "tier123_results_s1_end2end"),
        batch_size_verify=8,
        ece_bins=15,
        seed=13,
        # optional:
        # calib_max_steps=500,
        # calib_lr=0.05,
    )
    s1.run()

    ''' S3 Chexpert'''
    s3 = S3Tier123End2EndEvaluatorChexpertPlus(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        verifier_weights_path=verifier_weights_path,
        device="cuda",
        top_k=20,
        verifier_max_tokens=int(params["Verifier"]["max_tokens"]),
        out_dir=os.path.join(base_path_evidenceData, "external_validation", "chexpertplus", "tier123_results_s3_end2end"),
        batch_size_verify=8,
        ece_bins=15,
        seed=13,
        bm25_k1=1.2,
        bm25_b=0.75,
        # optional speed knobs
        # faithfulness_max_eval=2000,
        # faithfulness_random_trials=1,
        # calibration (recommended for external validation: none)
        calibration_method="none",
        # calibration_method="temperature_on_test",  # not recommended
        # calib_max_steps=500,
        # calib_lr=0.05,
    )
    s3.run()

    ''' S2 Chexpert'''
    s2 = S2Tier123End2EndEvaluatorChexpertPlus(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        verifier_weights_path=verifier_weights_path,
        device="cuda",
        top_k=20,
        verifier_max_tokens=int(params["Verifier"]["max_tokens"]),
        out_dir=os.path.join(base_path_evidenceData, "external_validation", "chexpertplus", "tier123_results_s2_end2end"),
        batch_size_embed_chunks=128,
        batch_size_retrieve=256,
        batch_size_verify=8,
        ece_bins=15,
        seed=13,
        # optional speed knobs
        # faithfulness_max_eval=2000,
        # faithfulness_random_trials=1,
        # calibration (recommended: none for external validation)
        calibration_method="none",
        # calibration_method="temperature_on_test",  # not recommended
        # calib_max_steps=500,
        # calib_lr=0.05,
    )
    s2.run()

    ''' S4 Chexpert'''
    runner = S4Tier123End2EndEvaluatorChexpertPlus(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        model_weights_path=model_weights_path,
        verifier_weights_path=verifier_weights_path,
        device="cuda",
        top_k=20,
        verifier_max_tokens=int(params["Verifier"]["max_tokens"]),
        out_dir=os.path.join(base_path_evidenceData, "external_validation", "chexpertplus", "tier123_results_s4_end2end"),
        batch_size_retrieve=256,
        batch_size_verify=8,
        ece_bins=15,
        seed=13,
        # Tier2 knobs
        # faithfulness_max_eval=2000,
        # faithfulness_random_trials=1,
        # Tier3 knobs
        calibration_method="none",  # recommended for external validation
        # calibration_method="temperature",  # fits on test (not recommended)
        calibration_opt_steps=200,
        calibration_lr=0.05,
    )
    runner.run()


    ''' Cross comparison Chexpert'''
    base = params["Preprocessing"]["base_path_evidenceData"]

    base_out = os.path.join(base, "external_validation", "chexpertplus")

    comparator = CrossSystemComparatorPredOnlyChexpertPlus(
        system_test_predictions={
            "S1": os.path.join(base_out, "tier123_results_s1_end2end", "s1_test_predictions.parquet"),
            "S2": os.path.join(base_out, "tier123_results_s2_end2end", "s2_test_predictions.parquet"),
            "S3": os.path.join(base_out, "tier123_results_s3_end2end", "s3_test_predictions.parquet"),
            "S4": os.path.join(base_out, "tier123_results_s4_end2end", "s4_test_predictions.parquet"),
        },
        system_faithfulness_predictions={
            "S2": os.path.join(base_out, "tier123_results_s2_end2end", "s2_tier2_faithfulness_predictions.parquet"),
            "S3": os.path.join(base_out, "tier123_results_s3_end2end", "s3_tier2_faithfulness_predictions.parquet"),
            "S4": os.path.join(base_out, "tier123_results_s4_end2end", "s4_tier2_faithfulness_predictions.parquet"),
        },
        out_path=os.path.join(
            base_out,
            "cross_system_results",
            "s1_s4_cross_system_comparison_predonly_chexpertplus.json",
        ),
        ece_bins=15,
        n_boot=2000,
        seed=13,
    )
    comparator.run()





def main_vignette_evaluation(global_config_path, experiment_name, model_weights_path, verifier_weights_path):

    with open(global_config_path, "r") as f:
        params = yaml.safe_load(f)
    base_path_mimicData = params["Preprocessing"]["base_path_mimicData"]
    base_path_chexpertData = params["Preprocessing"]["base_path_mimicData"]
    base_path_evidenceData = params["Preprocessing"]["base_path_evidenceData"]


    # data processing
    csv_4 = os.path.join(base_path_mimicData, "Vignette_4option.csv")
    csv_5 = os.path.join(base_path_mimicData, "Vignette_5option.csv")
    processor = RadiologyVignetteProcessor(retrieval_query_mode="context_plus_claim")
    artifacts = processor.build_from_csvs(csv_paths=[csv_4, csv_5], dataset_name="radiology169" )
    questions_df = artifacts["questions_df"]  # 1 row per question
    expanded_df = artifacts["expanded_df"]  # 1 row per (question, option)
    print("Questions:", len(questions_df))
    print("Expanded rows:", len(expanded_df))
    out_paths = processor.save_artifacts(
        out_dir= base_path_mimicData,
        artifacts=artifacts,
        prefix="radiology169")
    print(out_paths)




    ''' S4 Vignette '''
    runner = VignetteS4Tier123End2EndEvaluator(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        model_weights_path=model_weights_path,
        verifier_weights_path=verifier_weights_path,
        vignette_paths=VignetteEvalPaths(
            csv_4option=os.path.join(base_path_chexpertData, "Vignette_4option.csv"),
            csv_5option=os.path.join(base_path_chexpertData, "Vignette_5option.csv"),
        ),
        device="cuda",
        top_k=20,
        verifier_max_tokens=int(params["Verifier"]["max_tokens"]),
        batch_size_retrieve=256,
        batch_size_verify=4,
        ece_bins=15,
        seed=13,
        faithfulness_random_trials=1,
        out_dir=os.path.join(base_path_evidenceData, "external_validation", "Vignette", "tier123_results_s4_end2end"),
    )
    runner.run()

    ''' S3 Vignette '''
    s3 = VignetteS3Tier123End2EndEvaluator(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        verifier_weights_path=verifier_weights_path,
        vignette_paths=VignetteEvalPaths(
            csv_4option=os.path.join(base_path_chexpertData, "Vignette_4option.csv"),
            csv_5option=os.path.join(base_path_chexpertData, "Vignette_5option.csv")),
        device="cuda",
        top_k=20,
        verifier_max_tokens=int(params["Verifier"]["max_tokens"]),
        batch_size_verify=4,
        ece_bins=15,
        seed=13,
        bm25_k1=1.2,
        bm25_b=0.75,
        # tier2 speed knobs
        # faithfulness_max_eval_questions=100,
        # faithfulness_random_trials=1,
        out_dir=os.path.join(base_path_evidenceData, "external_validation", "Vignette", "tier123_results_s3_end2end"),
    )
    s3.run()

    ''' S2 Vignette '''
    s2 = VignetteS2Tier123End2EndEvaluator(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        verifier_weights_path=verifier_weights_path,
        vignette_paths=VignetteEvalPaths(
            csv_4option=os.path.join(base_path_chexpertData, "Vignette_4option.csv"),
            csv_5option=os.path.join(base_path_chexpertData, "Vignette_5option.csv")),
        device="cuda",
        top_k=20,
        verifier_max_tokens=int(params["Verifier"]["max_tokens"]),
        batch_size_embed_chunks=128,
        batch_size_retrieve=256,
        batch_size_verify=4,
        ece_bins=15,
        seed=13,
        # tier2 speed knobs
        # faithfulness_max_eval_questions=100,
        # faithfulness_random_trials=1,
        out_dir=os.path.join(base_path_evidenceData, "external_validation", "Vignette", "tier123_results_s2_end2end"),
    )
    s2.run()

    ''' S1 Vignette '''
    s1 = VignetteS1Tier123End2EndEvaluator(
        cfg_path=global_config_path,
        experiment_name=experiment_name,
        base_path_evidenceData=base_path_evidenceData,
        verifier_weights_path=verifier_weights_path,
        vignette_paths=VignetteEvalPaths(
            csv_4option=os.path.join(base_path_chexpertData, "Vignette_4option.csv"),
            csv_5option=os.path.join(base_path_chexpertData, "Vignette_5option.csv")),
        device="cuda",
        verifier_max_tokens=int(params["Verifier"]["max_tokens"]),
        batch_size_verify=4,
        ece_bins=15,
        seed=13,
        out_dir=os.path.join(base_path_evidenceData, "external_validation", "Vignette", "tier123_results_s1_end2end"),
    )
    s1.run()


    ''' Cross comparison Vignette'''
    base = params["Preprocessing"]["base_path_evidenceData"]
    base_out = os.path.join(base, "external_validation", "Vignette")

    comparator = CrossSystemComparatorPredOnlyVignette(
        system_question_predictions={
            "S1": os.path.join(base_out, "tier123_results_s1_end2end", "s1_question_level_predictions_uncalibrated.parquet"),
            "S2": os.path.join(base_out, "tier123_results_s2_end2end", "s2_question_level_predictions_uncalibrated.parquet"),
            "S3": os.path.join(base_out, "tier123_results_s3_end2end", "s3_question_level_predictions_uncalibrated.parquet"),
            "S4": os.path.join(base_out, "tier123_results_s4_end2end", "question_level_predictions_uncalibrated.parquet"),
        },
        system_tier2_question_predictions={
            "S1": os.path.join(base_out, "tier123_results_s1_end2end", "s1_tier2_input_perturbations_predictions_question_level.parquet"),
            "S2": os.path.join(base_out, "tier123_results_s2_end2end", "s2_tier2_faithfulness_predictions_question_level.parquet"),
            "S3": os.path.join(base_out, "tier123_results_s3_end2end", "s3_tier2_faithfulness_predictions_question_level.parquet"),
            "S4": os.path.join(base_out, "tier123_results_s3_end2end", "tier2_faithfulness_predictions_question_level.parquet"),
        },
        out_path=os.path.join(
            base_out,
            "cross_system_results",
            "s1_s4_cross_system_comparison_vignette.json",
        ),
        ece_bins=15,
        n_boot=2000,
        seed=13,
    )
    comparator.run()










if __name__ == '__main__':

    with open(global_config_path, "r") as f:
        params = yaml.safe_load(f)
    base_path_evidenceData = params["Preprocessing"]["base_path_evidenceData"]


    # main_train_retriever(global_config_path=global_config_path, valid=True, resume=False, experiment_name='name')

    main_train_verifier(global_config_path=global_config_path, valid=True, resume=False, experiment_name="nameverifier")


    end2end_mimic_processing_4_7(global_config_path=global_config_path, experiment_name=experiment_name, model_weights_path=model_weights_path)

    end2end_chexpert_processing_4_7(global_config_path=global_config_path, experiment_name=experiment_name, model_weights_path=model_weights_path)

    main_mimic_evaluation(global_config_path=global_config_path, experiment_name=experiment_name,
                          model_weights_path=model_weights_path, verifier_weights_path=verifier_weights_path)

    main_chexpert_evaluation(global_config_path=global_config_path, experiment_name=experiment_name,
                          model_weights_path=model_weights_path, verifier_weights_path=verifier_weights_path)

