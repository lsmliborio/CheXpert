import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import glob


def get_data(csv_path, target):
    df = pd.read_csv(csv_path)
    target_df = df[['Path', target]][(df[target] == 0) | (df[target] == 1)]
    
    positive = target_df[target_df[target] == 1].drop(columns=[target])
    negative = target_df[target_df[target] == 0].drop(columns=[target])
    
    return positive, negative

def get_full_data(path):
    targets = np.array(['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                        'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                        'Fracture', 'Support Devices'])
    full_data = {}
    for target in targets:
        positive, negative = get_data(path, target)
        full_data[target] = [positive, negative]
    
    return full_data


def gen_episode(positive, negative, positive_shots=2, negative_shots=2, query_shots=4, batch_size=1):
    """
    take the dataset as input and outputs a few-shot batch episode
    inputs:
        positive/negative -> .csv filtered by class
        anom_shots  -> number of anomaly shots
        good_shots  -> number of template shots
        query_shots -> number of test shots
    outputs:
        train_batch -> a list with all inputs in order and batched
        labels -> the respective training labels of this episode
    """
    positive_indices  = np.arange(positive.shape[0])
    negative_indices  = np.arange(negative.shape[0])      
    
    positive_batch  = []
    negative_batch  = []
    query_batch = []
    labels      = []
    train_batch = []

    for i in range(batch_size):
        
        query_class = np.random.randint(0, 2) # random class for the query         
        
        # SUPPORT
        # positive shots
        positive_shots_indices = np.random.choice(
            positive_indices, positive_shots, replace=False
        )
        sample_positive_shots = positive.iloc[positive_shots_indices]
        # negative shots
        negative_shots_indices = np.random.choice(
                negative_indices, negative_shots, replace=False
        )
        sample_negative_shots = negative.iloc[negative_shots_indices]
        
        
        # QUERY
        positive_query_shots = query_shots // 2 # number of query shots separated for anomalies
        negative_query_shots = query_shots - positive_query_shots
        
        # do not take queries that are equal to the support set images
        positive_query_indices = np.delete(positive_indices, positive_shots_indices)
        negative_query_indices = np.delete(negative_indices, negative_shots_indices)
        
        positive_query_indices = np.random.choice(positive_query_indices, positive_query_shots)
        negative_query_indices = np.random.choice(negative_query_indices, negative_query_shots)
        
        
        sample_query_shots = np.concatenate([positive.iloc[positive_query_indices],
                                             negative.iloc[negative_query_indices]]
                                           )
        
        query_label = np.concatenate([np.ones(positive_query_shots),
                                      np.zeros(negative_query_shots)]
                                    )
        
        query_indices = np.arange(sample_query_shots.shape[0])
        np.random.shuffle(query_indices)
        sample_query_shots = sample_query_shots[query_indices] #.reshape(-1, *sample_query_shots.shape[-3:])
        query_label = query_label[query_indices]
        
        
        # generate episode labels
        label_1 = np.array((1 == query_label) * 1)
        label_2 = np.array((0 == query_label) * 1)
        ep_label = np.column_stack([label_1, label_2])
                
        # few-shot batches
        positive_batch.append(sample_positive_shots)
        negative_batch.append(sample_negative_shots)
        query_batch.append(sample_query_shots)
        labels.append(ep_label)
    
    # organize each few-shot batch into a single array of shape == (shots, batch_size, *img.shape)
    train_batch = []
    for i in range(positive_shots):
        train_batch.append(np.array(positive_batch)[:, i])
    for i in range(negative_shots):
        train_batch.append(np.array(negative_batch)[:, i])
    for i in range(query_shots):
        train_batch.append(np.array(query_batch)[:, i])
    
    # organize batch label
    labels = np.array(labels)
    labels = labels.reshape((-1, *labels.shape[1:]))
    
    
    episode_imgs = []    
    for shot in train_batch:
        batch_imgs = []        
        for batch in shot:            

            img_path = batch[0]
            img = cv2.imread(img_path, 0)
            img = cv2.resize(img, (320, 320))
            img = np.expand_dims(img, -1)

            batch_imgs.append(img / 255)

        episode_imgs.append(np.array(batch_imgs).astype(np.float32))    
    
#     labels[labels == 0] = -1
    return episode_imgs, labels


def fetch_RN(positive,
             negative,
             positive_shots, 
             negative_shots,
             query_shots,
             batch_size):
    
    X, y = gen_episode(positive, negative,
                       positive_shots,
                       negative_shots,
                       query_shots,
                       batch_size)

    return X, y