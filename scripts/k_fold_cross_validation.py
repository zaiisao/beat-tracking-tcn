"""
Ben Hayes 2020

ECS7006P Music Informatics

Coursework 1: Beat Tracking

File: scripts/k_fold_cross_validation.py

Descrption: Train several BeatNet models on different folds of the dataset
            and compare their performance.
"""
from argparse import ArgumentParser
import math
import os

from torch import device, save
from torch.utils.data import random_split, ConcatDataset

from beat_tracking_tcn.models.beat_net import BeatNet
from train import test_model,\
                   train_loop,\
                   save_model,\
                   make_data_loaders,\
                   split_dataset,\
                   load_dataset


def parse_args():
    parser = ArgumentParser(
        description="Perform k-fold cross-validation on a BeatNet model")

    parser.add_argument("spectrogram_dir", type=str)
    parser.add_argument("label_dir", type=str)
    parser.add_argument(
        "-k",
        "--num_folds",
        default=8,
        type=str,
        help="Number of folds for cross-validation")
    parser.add_argument(
        "-o",
        "--output_file",
        default=None,
        type=str,
        help="Where to save trained model.")
    parser.add_argument(
        "-n",
        "--num_epochs",
        default=100,
        type=int,
        help="Number of training epochs")
    parser.add_argument(
        "-s",
        "--davies_stopping_condition",
        action="store_true",
        help="Use Davies & Bock's stopping condition " +
             "(ignores number of epochs)")
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,
        help="Batch size to use for training.")
    parser.add_argument(
        "-c",
        "--cuda_device",
        type=int,
        default=None,
        help="CUDA device index for training. CPU used if none specified.")
    parser.add_argument(
        "-d",
        "--dataset_output_file",
        type=str,
        default=None,
        help="Save directory for datasets to allow for consistent evaluation")
    parser.add_argument(
        "--downbeats",
        action="store_true",
        help="Trains a downbeat tracking model")

    return parser.parse_args()


def make_fold_datasets(dataset, num_folds):
    """
    Split datasets into folds.
    """
    dataset_length = len(dataset)
    fold_size = math.floor(dataset_length / num_folds)
    remainder = dataset_length - (num_folds * fold_size)

    fold_counts = [fold_size] * num_folds
    for i in range(remainder):
        fold_counts[i] += 1

    fold_sets = random_split(dataset, fold_counts) #MJ: List of Subsets of  dataset at specified indices.

    return fold_sets

def iterate_folds(fold_sets):
    """
    Do necessary array slices to iterate correctly over folds for training,
    validating and testing.
    """
    num_sets = len(fold_sets)
    for i in range(num_sets): #MJ: num_sets = 8
        test_set = fold_sets[i]
        val_set = fold_sets[(i + 1) % num_sets]
        train_sets = fold_sets[:i] + fold_sets[(i + 2):]
        train_set = ConcatDataset(train_sets)

        yield train_set, val_set, test_set
        
#MJ:  output_file = make_fold_output_name(args.output_file, k)
def make_fold_output_name(base_name, fold):
    """
    Add numbers to output file name.
    """
    filename, ext = os.path.splitext(base_name)
    new_name = "%s.fold%.3d%s" % (filename, fold, ext)
    return new_name

def save_datasets(datasets, file):
    """
    Dump with PyTorch save
    """    
    with open(file, 'wb') as f:
        save(datasets, f)

#MJ: Davies & Boek, 2019:  To permit the use of existing annotated datasets, both to train our model and measure its
# performance, we use 8-fold cross validation. We ensure the separation between testing and training data 
# in each iteration of the cross validation, by using six folds for training, one for validation, 
# and the remaining fold for testing, and rotate the folds eight times 
# until each fold has uniquely been used for testing. 
if __name__ == "__main__":
    args = parse_args()
    
    dataset = load_dataset(
        args.spectrogram_dir,
        args.label_dir,
        args.downbeats)
    
    fold_sets = make_fold_datasets(dataset, args.num_folds)

    cuda_device = device('cuda:%d' % args.cuda_device)\
                  if args.cuda_device is not None else None

    # JA: Similar to our train script, but we do this k times
    for k, datasets in enumerate(iterate_folds(fold_sets)):
        
        train, val, test = datasets
        
        model = BeatNet(downbeats=args.downbeats)
        if cuda_device is not None:
            model.cuda(args.cuda_device)

        output_file = make_fold_output_name(args.output_file, k)

        train_loader, val_loader, test_loader = make_data_loaders(
            (train, val, test),
            batch_size=args.batch_size)

        train_loop(
            model,
            train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            cuda_device=cuda_device,
            output_file=output_file,
            davies_stopping_condition=args.davies_stopping_condition,
            
            fold=k)

        if args.output_file is not None:
            save_model(model, output_file)
        
        if args.dataset_output_file is not None:
            save_dir = make_fold_output_name(args.dataset_output_file, k)
            save_datasets((train, val, test), save_dir)

        test_model(model, test_loader, cuda_device=cuda_device)