import torch
import torchvision.transforms as transforms
import timm
from tqdm import tqdm
from torch.utils.data import Dataset, Subset, DataLoader, Sampler
import glob
import os
from PIL import Image
from collections import defaultdict, Counter
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import seaborn as sns
from sklearn.manifold import TSNE
import json
import math
from datasets import load_dataset

os.environ["LOKY_MAX_CPU_COUNT"] = "6"

SEED = 42

class custom_dataset(Dataset):
    def __init__(self, vanila_dataset, image_indexes, transform):
        self.dataset = vanila_dataset
        self.image_indexes = image_indexes

        self.transform = transform

    def __len__(self):
        return len(self.image_indexes)
    
    def __getitem__(self, index):
        selected_index = self.image_indexes[index]
        
        _image = self.dataset[selected_index]['img']
        _image = self.transform(_image)

        _label = self.dataset[selected_index]['label']

        return _image, _label, selected_index
    

def worker_init_fn(worker_id):
    seed = worker_id + SEED

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    return


class custom_data_factory():
    def __init__(self, train_dataset, test_dataset, sample_num, train_transform, valid_transform):
        self.train_data_dict = {}

        for i in range(0, 50000):
            _index = i
            _label = train_dataset[i]['label']
            if self.train_data_dict.get(_label) is None:
                self.train_data_dict[_label] = []

            self.train_data_dict[_label].append(_index)

        self.train_datset = train_dataset
        self.test_dataset = test_dataset

        self.selected_items = {_label: random.choices(item_index, k=min(len(item_index), sample_num)) for _label, item_index in self.train_data_dict.items()}

        self.sample_num = sample_num
        self.train_transform = train_transform
        self.valid_transform = valid_transform

        self.losses = []

    def get_data_set(self):
        _item_list =  sum(self.selected_items.values(), [])
        random.shuffle(_item_list)
        
        self.train_set = custom_dataset(self.train_datset, _item_list, self.train_transform)
        self.valid_set = custom_dataset(self.test_dataset, [i for i in range(0, 10000)], self.valid_transform)

        print(f'Train set: {len(self.train_set.image_indexes)}')
        print(f'Valid set: {len(self.valid_set.image_indexes)}')

        return self.train_set, self.valid_set
    
    def get_data_loader(self):
        train_loader = DataLoader(
            self.train_set, pin_memory=True,
            batch_size=16,
            num_workers=0,
        )

        valid_loader = DataLoader(
            self.valid_set, pin_memory=True,
            batch_size=16,
            num_workers=0,
        )

        return train_loader, valid_loader
    
    def collect_losses(self, _paths, _labels, _losses):
        for _path, _label, _loss in zip(_paths, _labels, _losses):
            self.losses.append((_path, _label.item(), _loss.item()))

    def renew_data_loader(self, resample_ratio):
        selected_items_dict = {}
        for _path, _label, _loss in self.losses:
            if selected_items_dict.get(_label) == None:
                selected_items_dict[_label] = []

            selected_items_dict[_label].append((_path, _loss))
        
        selected_items = {}
        for _label, _path_list in selected_items_dict.items():
            _paths, _weights = zip(*_path_list)
            _weights = [(_weight - min(_weights))/(max(_weights)-min(_weights)) for _weight in _weights]

            selected_items[_label] = random.choices(_paths,
                                                    k=int(self.sample_num * (1-resample_ratio)),
                                                    weights=_weights)
        
        unselected_items = {_label: random.sample([_path for _path in _path_list if _path not in selected_items_dict[_label]],
                                                  k=int(self.sample_num * resample_ratio)) for _label, _path_list in self.train_data_dict.items()}

        self.selected_items = {}
        for (_label, _selected_path_list), (_label, _unselected_path_list) in zip(selected_items.items(), unselected_items.items()):
            self.selected_items[_label] = []
            self.selected_items[_label].extend(_selected_path_list)
            self.selected_items[_label].extend(_unselected_path_list)

        self.losses = []

        self.get_data_set()

        return self.get_data_loader()

def evaluate(model, valid_loader):
    model.eval()
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for images, labels, selected_indexes in tqdm(valid_loader, desc=f'Validating', leave=False):
            images, labels = images.to('cuda'), labels.to('cuda')
            
            outputs = model(images)

            probs = F.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy().flatten())
            all_labels.append(labels.cpu().numpy())
        
    all_probs_roc = np.array(all_probs).reshape(-1, 10)
    all_probs = np.array([np.argmax(_pred) for _pred in all_probs_roc])
    all_labels = np.array(all_labels).flatten()

    return {
        'accuracy': accuracy_score(all_labels, all_probs),
        'precision': precision_score(all_labels, all_probs, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_probs, average='weighted', zero_division=0),
        'f1': f1_score(all_labels, all_probs, average='weighted', zero_division=0),
        'roc_auc':roc_auc_score(all_labels, all_probs_roc, multi_class='ovr'),
        'confusion_matrix': confusion_matrix(all_labels, all_probs)
    }



if __name__ == '__main__':
    cifar_dataset = load_dataset("uoft-cs/cifar10")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                            std=[0.5, 0.5, 0.5])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                            std=[0.5, 0.5, 0.5])
    ])