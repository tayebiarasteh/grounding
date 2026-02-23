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
import timm
import numpy as np
from torch import nn
from sklearn import metrics
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from transformers import AutoImageProcessor, AutoModel
from opacus.validators import ModuleValidator

from config.serde import open_experiment, create_experiment, delete_experiment, write_config
from Train_Valid_grounding import Training
from Prediction_grounding import Prediction
from data.data_provider import RadiopaediaChunksStore, RetrieverSupervisionDataset, retriever_collate_fn
from models.model import BiEncoder, WeightedInfoNCELoss

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







def main_test_retriever(demographics, global_config_path="/PATH/dp_dinov3/config/config.yaml",
                                                 experiment_name='central_exp_for_test', experiment_epoch_num=100,
                        dataset_name='vindr', dino=True, image_size=224, new_seed=False):
    """Main function for multi label prediction

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']
    login(token=params["hf_login"])

    if dataset_name == 'vindr':
        test_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size, demographics=demographics)
    elif dataset_name == 'chexpert':
        test_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size, demographics=demographics)
    elif dataset_name == 'mimic':
        test_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size, demographics=demographics)
    elif dataset_name == 'UKA':
        test_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size, demographics=demographics)
    elif dataset_name == 'cxr14':
        test_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size, demographics=demographics)
    elif dataset_name == 'padchest':
        test_dataset = padchest_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False, image_size=image_size, demographics=demographics)
    weight = test_dataset.pos_weight()
    label_names = test_dataset.chosen_labels


    # Changeable network parameters
    if dino:
        model = AutoModel.from_pretrained(
            "facebook/dinov3-convnext-small-pretrain-lvd1689m",  # dinov3
            # "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",  # dinov3
            use_safetensors=True
        )
        model.head = torch.nn.Linear(in_features=768, out_features=len(weight))
    else:
        model = AutoModel.from_pretrained(
            "facebook/convnext-small-224", # ImageNet-1k
            # "facebook/convnext-tiny-224", # ImageNet-1k
            use_safetensors=True
        )
        model.head = torch.nn.Linear(in_features=768, out_features=len(weight))


    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=11)


    os.makedirs(os.path.join(params['target_dir'], params['stat_log_path'], demographics), exist_ok=True)
    if new_seed:
        index_list = []
        for counter in range(1000):
            index_list.append(np.random.choice(len(test_dataset), len(test_dataset)))
        np.save(os.path.join(params['target_dir'], params['stat_log_path'], demographics) + '/bootstrapping_seeds_' + str(dataset_name) + ".npy", index_list, allow_pickle=True)


    else:
        index_list = np.load(os.path.join(params['target_dir'], params['stat_log_path'], demographics) + '/bootstrapping_seeds_' + str(dataset_name) + ".npy", allow_pickle=True)

    # Initialize prediction
    predictor = Prediction(cfg_path, label_names)
    predictor.setup_model(model=model, epoch_num=experiment_epoch_num)
    pred_array, target_array = predictor.predict_only(test_loader)

    #########################################
    pred_array = pred_array.cpu().numpy()
    target_array = target_array.int().cpu().numpy()
    np.save(os.path.join(params['target_dir'], params['stat_log_path'], demographics) + '/pred_array_'  + str(dataset_name) + '.npy', pred_array, allow_pickle=True)
    np.save(os.path.join(params['target_dir'], params['stat_log_path'], demographics) + '/target_array_'  + str(dataset_name) + '.npy', target_array, allow_pickle=True)

    df = pd.DataFrame(pred_array.mean(1), columns=['probability_mean'])
    for idx in range(pred_array.shape[-1]):
        df.insert(idx + 1, 'prob_' + label_names[idx], pred_array[:, idx])
        df.insert(idx + 1, 'gt_' + label_names[idx], target_array[:, idx])
    df.to_csv(os.path.join(params['target_dir'], params['stat_log_path'], demographics) + '/predictions_teston_' + str(
        dataset_name) + '.csv', sep=',', index=False)


    msg = (
        f'\n\n\n\t Epoch number: {experiment_epoch_num} below:'
    )
    with open(os.path.join(params['target_dir'], params['stat_log_path'], demographics) + '/Test_on_' + str(dataset_name), 'a') as f:
        f.write(msg)

    AUC_list = predictor.bootstrapper(pred_array, target_array, index_list, dataset_name, demographics=demographics)







if __name__ == '__main__':
    global_config_path = "/PATH/grounding/config/config.yaml"
    # delete_experiment(experiment_name="name", global_config_path=global_config_path)

    main_train_retriever(global_config_path=global_config_path, valid=True, resume=False, experiment_name='name')
