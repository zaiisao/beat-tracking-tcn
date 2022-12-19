"""
Ben Hayes 2020

ECS7006P Music Informatics

Coursework 1: Beat Tracking

File: scripts/train.py

Descrption: Train a BeatNet model on a given dataset.
"""
from argparse import ArgumentParser

import numpy as np
from torch.utils.data import random_split, DataLoader
from torch.nn import BCELoss
from torch.optim import Adam, lr_scheduler  #MJ: lr_scheduler.py
from torch import device, save

from beat_tracking_tcn.datasets.ballroom_dataset import BallroomDataset
from beat_tracking_tcn.models.beat_net import BeatNet
from beat_tracking_tcn.utils.training import train, evaluate


# Some tunable constants that we don't need to set often enough to
# warrant entire command line params
#MJ: We reduce the learn rate by a factor of 5 if the loss on the disjoint validation set reaches a plateau
# and stop training if no improvement in the validation loss is observed for 50 epochs.
STOPPING_THRESHOLD = 0.001
DAVIES_CONDITION_EPOCHS = 50


def parse_args():
    parser = ArgumentParser(
        description="Train a BeatNet model on a given dataset.")

    parser.add_argument("spectrogram_dir", type=str)
    parser.add_argument("label_dir", type=str)
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
        "-v",
        "--validation_split",
        type=float,
        default=0.1,
        help="Proportion of the data to use for validation.")
    parser.add_argument(
        "-t",
        "--test_split",
        type=float,
        help="Proportion of the data to use for testing.")
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


def load_dataset(spectrogram_dir, label_dir, downbeats=False):
    """
    Creates an instance of BallroomDataset from the given folders of
    spectrograms and labels.
    """    
    dataset = BallroomDataset(spectrogram_dir, label_dir, downbeats=downbeats)
    return dataset


def split_dataset(dataset, validation_split, test_split):
    """
    Splits a given torch.utils.data.Dataset into train, validation and test
    sets based on the given proportions.
    """    
    dataset_length = len(dataset)
    
    test_count = int(dataset_length * test_split)\
        if test_split is not None else 0
        
    val_count = int(dataset_length * validation_split)
    
    train_count = dataset_length - (test_count + val_count)
    #MJ: lengths = (train_count, val_count, test_count) = (4 4 3)
    return random_split(dataset, (train_count, val_count, test_count))


def make_data_loaders(datasets, batch_size=1, num_workers=8):
    """
    Given an iterable container of datasets, output a tuple of DataLoaders
    """    
    loaders = (
        DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        for dataset in datasets)
    return loaders


def save_model(model, output_file):
    """
    Dump the model state dict to disk using torch.save
    """
    state_dict = model.state_dict()
    with open(output_file, 'wb') as f:
        save(state_dict, f)


def save_datasets(datasets, file):
    """
    Dump the datasets to disk using torch.save
    """    
    with open(file, 'wb') as f:
        save(datasets, f)


def loss_stopped_falling(loss_history, epochs): #MJ: epochs = DAVIES_CONDITION_EPOCHS = 50
    """
    Check if drop in first order difference over a given number of epochs is
    below the STOPPING_THRESHOLD.
    """    
    return - np.sum( np.diff( loss_history[-epochs:] ) ) < STOPPING_THRESHOLD


def train_loop(
        model,
        train_loader,
        val_loader=None,
        num_epochs=100,
        learning_rate=0.001,
        cuda_device=None,
        output_file=None,
        davies_stopping_condition=False,
        fold=None):
    """
    Run the main training loop.
    """    

    # The train function defined in the beat_tracking_tcn library offloads
    # reporting to callbacks, which are implemented here:
    def train_callback(batch_report):
        if batch_report["batch_index"] % 10 == 9:
            if fold is None:
                print("Training Batch %d; Loss: %.3f; Epoch Loss: %.3f" % (
                        batch_report["batch_index"],
                        batch_report["batch_loss"],
                        batch_report["running_epoch_loss"]), end="\r")
            else:
                print("Fold %d; Training Batch %d; Loss: %.3f; Epoch Loss: %.3f" % (
                        fold,
                        batch_report["batch_index"],
                        batch_report["batch_loss"],
                        batch_report["running_epoch_loss"]), end="\r")
    
    def val_callback(batch_report):
        if batch_report["batch_index"] % 10 == 9:
            if fold is None:
                print("Validation Batch %d; Loss: %.3f; Epoch Loss: %.3f" % (
                        batch_report["batch_index"],
                        batch_report["batch_loss"],
                        batch_report["running_epoch_loss"]), end="\r")
            else:
                print("Fold: %d; Validation Batch %d; Loss: %.3f; Epoch Loss: %.3f" % (
                        fold,
                        batch_report["batch_index"],
                        batch_report["batch_loss"],
                        batch_report["running_epoch_loss"]), end="\r")

    val_loss_history = []
    
    criterion = BCELoss()
    optimiser = Adam(model.parameters(), lr=learning_rate)
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.2)

    #MJ: within train_loop():
    for epoch in range(num_epochs):
        
        epoch_report = train(
            model,
            criterion,
            optimiser,
            train_loader,
            batch_callback=train_callback,
            cuda_device=cuda_device)

        if val_loader is not None:
            
            val_report = evaluate(
                model,
                criterion,
                val_loader,
                batch_callback=val_callback,
                cuda_device=cuda_device)
            
            #MJ: Use scheduler to adjust lr: class ReduceLROnPlateau:
            scheduler.step(val_report["epoch_loss"])

            val_loss_history.append(val_report["epoch_loss"])

            if fold is None:
                print("Epoch #%d; Loss: %.3f; Val Loss: %.3f                   " % (
                        epoch,
                        epoch_report["epoch_loss"],
                        val_report["epoch_loss"]))
            else:
                print("Fold: #%d; Epoch #%d; Loss: %.3f; Val Loss: %.3f                   " % (
                        fold,
                        epoch,
                        epoch_report["epoch_loss"],
                        val_report["epoch_loss"]))
            
            if davies_stopping_condition:
                if loss_stopped_falling(
                        val_loss_history,
                        DAVIES_CONDITION_EPOCHS)\
                and len(val_loss_history) > DAVIES_CONDITION_EPOCHS:
                    break
        else:
            if fold is None:
                print("Epoch #%d; Loss: %.3f                                    " %
                    (epoch, epoch_report["epoch_loss"]))
            else:
                print("Fold: #%d; Epoch #%d; Loss: %.3f                                    " %
                    (fold, epoch, epoch_report["epoch_loss"]))

        
        if output_file is not None:
            save_model(model, output_file)

    return model


def test_model(model, test_loader, cuda_device=None):
    """
    Evaluate the model on the dataset slice pointed to by the given DataLoader.
    """    
    def test_callback(batch_report):
        print("Test Batch %d; Loss: %.3f; Epoch Loss: %.3f" % (
                batch_report["batch_index"],
                batch_report["batch_loss"],
                batch_report["running_epoch_loss"]), end="\r")
    
    criterion = BCELoss()

    #MJ: evaluate():   return {
    #     "total_batches": i + 1,
    #     "epoch_loss": running_loss / (i + 1),
    #     "running_evaluations": running_evaluations 
    # }
    test_report = evaluate(
        model,
        criterion,
        test_loader,
        batch_callback=test_callback,
        cuda_device=cuda_device)
    
    print("Test Loss: %.5f                                                 " %
          test_report["epoch_loss"])


if __name__ == '__main__':
    
    args = parse_args()

    # Make sure we're not trying to do anything silly like use the stopping
    # condition with a validation size of 0
    if args.validation_split == 0.0 and args.davies_stopping_condition:
        print("Validation split must be greater than zero in order to use "
              + "Davies stopping condition.")
        quit()

    # Prepare datasets and DataLoaders
    dataset = load_dataset(
        args.spectrogram_dir,
        args.label_dir,
        args.downbeats)
    
    train_dataset, val_dataset, test_dataset =\
        split_dataset(dataset, args.validation_split, args.test_split)
        
    train_loader, val_loader, test_loader =\
        make_data_loaders(
            (train_dataset, val_dataset, test_dataset),
            batch_size=args.batch_size)

    # Initialise model with GPU acceleration if possible
    cuda_device = device('cuda:%d' % args.cuda_device)\
                  if args.cuda_device is not None else None
    model = BeatNet(downbeats=args.downbeats)
    if cuda_device is not None:
        model.cuda(args.cuda_device)

    # Kick off training
    train_loop(
        model,
        train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        cuda_device=cuda_device,
        output_file=args.output_file,
        davies_stopping_condition=args.davies_stopping_condition)  #MJ: fold = None; k_fold_cross_validation.py uses fold = k

    # Save our model to disk 
    if args.output_file is not None:
        save_model(model, args.output_file)

    # Save our dataset splits for reproducibility
    if args.dataset_output_file is not None:
        save_datasets(
            (train_dataset, val_dataset, test_dataset),
            args.dataset_output_file)

    # Evaluate on our test set
    test_model(model, test_loader, cuda_device=cuda_device)  #MJ: fold = None
