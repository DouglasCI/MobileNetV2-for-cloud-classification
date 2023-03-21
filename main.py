import os
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import time
import copy
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.optim import Adam
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

TRAIN_SPLIT = 0.7
IMG_SIZE = (224, 224)

def get_loaders(root, batch_size, num_workers):
    """
    Parameters:
        root (str): path to directory containing dataset.
        batch_size (int): size of batch.
        num_workers (int): assign [num_workers] subprocesses for data loading.

    Returns:
        dict: dictionary containing train and validation dataloaders.
        DataLoader: test dataloader.
    """
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=IMG_SIZE),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    train_val_dataset = datasets.ImageFolder(root=os.path.join(root, 'train'), transform=preprocess)
    test_dataset = datasets.ImageFolder(root=os.path.join(root, 'test'), transform=preprocess)
    
    # stratified split train dataset into train and validation parts
    targets = train_val_dataset.targets
    train_idx, val_idx = train_test_split(np.arange(len(targets)), train_size=TRAIN_SPLIT, 
                                        shuffle=True, stratify=targets)
    
    train_dataset = Subset(train_val_dataset, train_idx)
    val_dataset = Subset(train_val_dataset, val_idx)
    
    # create DataLoaders for the datasets
    train_loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers), 
        'val': DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)}
    test_loader = DataLoader(test_dataset, batch_size=57,
                            shuffle=False, num_workers=num_workers)
    
    print('> Loaded {} train images, {} validation images and {} test images from {}.'.format(
        len(train_loaders['train'].dataset), len(train_loaders['val'].dataset),
        len(test_loader.dataset), root))
    
    return train_loaders, test_loader

def augment_batch(batch):
    """
    Parameters:
        batch (tensor): tensor containing data from this batch.

    Returns:
        tensor: augmented batch.
    """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(size=IMG_SIZE, pad_if_needed=True, padding_mode='reflect'),
        transforms.RandomRotation(degrees=(-15, 15))])
    
    batch_aug = transform(batch)
    
    return batch_aug
        
def train(model, dataloaders, device, optimizer, loss_func, epochs, augment_data):
    """
    Parameters:
        model (class): model for training.
        dataloaders (DataLoader): dataloaders for training.
        device (class): device where training will happen.
        optimizer (class): optimizer.
        loss_func (class): loss function.
        epochs (int): number of epochs.
        augment_data (boolean): use runtime data augmentation?

    Returns:
        class: trained model.
        dict: statistics history.
    """
    
    # statistics history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []}
    
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    total_runtime = 0.0
    
    print('> Training...')
    
    for epoch in range(epochs):
        start = time.time()
        
        train_loss = 0.0
        train_corrects = 0
        train_acc = 0.0
        val_loss = 0.0
        val_corrects = 0
        val_acc = 0.0
        
        for phase in ['train', 'val']:
            model.train() if phase == train else model.eval()
            
            # counter used for data augmentation (only for train phase)
            counter = 1
            
            if phase == 'train' and augment_data:
                counter = 20
            
            for _ in range(counter):
                for batch_data, labels in dataloaders[phase]:
                    # data augmentation
                    if phase == 'train' and augment_data:
                        batch_data = augment_batch(batch_data)
                        
                    # send tensors to device
                    batch_data = batch_data.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        # get predictions
                        preds = model(batch_data)
                        
                        loss = loss_func(preds, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    # count correct predictions
                    preds = torch.argmax(preds, axis=1)
                    if phase == 'train':
                        train_loss += loss.item() * batch_data.size(0)
                        train_corrects += torch.sum(preds == labels.data)
                    else:
                        val_loss += loss.item() * batch_data.size(0)
                        val_corrects += torch.sum(preds == labels.data)

            dataset_size = len(dataloaders[phase].dataset)
            if phase == 'train':
                train_loss /= dataset_size * counter
                train_acc = train_corrects.double() / (dataset_size * counter)
            else:
                val_loss /= dataset_size
                val_acc = val_corrects.double() / dataset_size
            
            # store best accuracy and model
            if phase == 'val' and val_acc > best_acc:
                best_acc = val_acc
                best_model = copy.deepcopy(model.state_dict())
            
            # clear GPU cache
            # torch.cuda.empty_cache()
            
        epoch_runtime = time.time() - start
        total_runtime += epoch_runtime
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.cpu().detach().numpy())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.cpu().detach().numpy())
        
        print('[EPOCH {}/{}] Train Loss: {:.4f} Acc: {:.4f} | Val Loss: {:.4f} Acc: {:.4f} | Runtime: {:.2f}s/epoch'.format(
            epoch + 1, epochs, train_loss, train_acc, val_loss, val_acc, epoch_runtime))
    
    print('> Total runtime: {:.0f}m {:.0f}s {:.0f}ms'.format(total_runtime // 60, total_runtime % 60, (total_runtime * 1000) % 1000))
    
    print('> Best validation accuracy was {:.2%}'.format(best_acc))
    
    model.load_state_dict(best_model)
    
    return model, history
         
def test(model, dataloader, device):
    """
    Parameters:
        model (class): trained model to be tested.
        dataloader (DataLoader): test dataloader.
        device (class): device where test will happen.
    """
    preds = []
    all_labels = dataloader.dataset.targets
    
    model.eval()
    
    with torch.no_grad():
        for batch_data, _ in dataloader:
            batch_data = batch_data.to(device)
            
            pred = model(batch_data)
            preds.extend(torch.argmax(pred, axis=1).cpu().numpy())
            
    classes_names = ["Ac", "As", "Cb", "Cc", "Ci", "Cs", "Ct", "Cu", "Ns", "Sc", "St"]
    print(classification_report(np.array(all_labels), np.array(preds), target_names=classes_names, zero_division=0))

def main():
    # training settings
    parser = argparse.ArgumentParser(description='Cloud Classification MobileNet')
    required_parser = parser.add_argument_group('required arguments')
    required_parser.add_argument('-dp', '--data-path', type=str, required=True, metavar='\b',
                        help='path to dataset directory')
    required_parser.add_argument('-m', '--model', type=str, required=True, metavar='\b',
                        help='model output filename')
    required_parser.add_argument('-p', '--plot', type=str, required=True, metavar='\b',
                        help='plot output filename')
    parser.add_argument('-bs', '--batch-size', type=int, default=22, metavar='\b',
                        help='batch size for training (default: 22)')
    parser.add_argument('-ep', '--epochs', type=int, default=10, metavar='\b',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('-lr', '--learn-rate', type=float, default=0.0001, metavar='\b',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training (default: False)')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training (default: False)')
    parser.add_argument('--freeze-top', action='store_true', default=False,
                        help='freeze only top layers')
    parser.add_argument('--augment-data', action='store_true', default=False,
                        help='choose to augment train data at runtime')
    parser.add_argument('--seed', type=int, default=1, metavar='\b',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    
    data_path = args.data_path
    batch_size = args.batch_size
    epochs = args.epochs
    learn_rate = args.learn_rate
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    num_workers = 0
    freeze_top = args.freeze_top
    augment_data = args.augment_data
    torch.manual_seed(args.seed)
    
    if use_cuda:
        device = torch.device('cuda')
        num_workers = 2
    elif use_mps:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    # get train and test datasets
    train_loaders, test_loader = get_loaders(data_path, batch_size, num_workers)

    # get MobileNet v2 model.
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights='IMAGENET1K_V2')
    
    # freeze layers
    if freeze_top:
        parameters = model.parameters()
        # freeze only 35 top layers
        for _ in range(35):
            next(parameters).requires_grad = False
    else:
        # freeze all layers
        for param in model.parameters():
            param.requires_grad = False
    
    # replace classifier layer with our own
    # this layer will not be frozen
    model.classifier = nn.Sequential(
        nn.Linear(in_features=model.classifier[1].in_features, out_features=300),
        nn.ReLU6(inplace=True),
        nn.Linear(in_features=300, out_features=200),
        nn.ReLU6(inplace=True),
        nn.Linear(in_features=200, out_features=512),
        nn.ReLU6(inplace=True),
        nn.Linear(in_features=512, out_features=11)
    )
        
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print('> Model created, it has {} trainable parameters and {} non trainable parameters'.format(
        trainable_params, non_trainable_params))
    
    model = model.to(device)
    
    optimizer = Adam(model.parameters(), lr=learn_rate)
    loss_func = nn.CrossEntropyLoss()
    
    # start training
    model, history = train(model, train_loaders, device, optimizer, loss_func, epochs, augment_data)
    
    # plot training statistics
    fig, axis = plt.subplots(2)
    fig.suptitle('Training x Validation')
    
    line1, = axis[0].plot(list(range(epochs)), history['train_loss'], label='train_loss')
    line2, = axis[0].plot(list(range(epochs)), history['val_loss'], label='val_loss')
    axis[0].set_ylabel('loss')
    axis[0].legend(handles=[line1, line2])
    
    line3, = axis[1].plot(list(range(epochs)), history['train_acc'], label='train_acc')
    line4, = axis[1].plot(list(range(epochs)), history['val_acc'], label='val_acc')
    axis[1].set_xlabel('epochs')
    axis[1].set_ylabel('accuracy')
    axis[1].legend(handles=[line3, line4])
    
    # save graph image and best model
    plt.savefig(args.plot)
    torch.save(model.state_dict(), args.model)
    
    # test with separated data
    test(model, test_loader, device)
    
if __name__ == '__main__':
    main()