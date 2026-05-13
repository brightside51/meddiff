import argparse
import sys
import torch

sys.path.append('../cfg')
from args import parser_def
sys.path.append('../dataset')
from brats import BRATSDataset
from mrnet import MRNetDataset
from adni import ADNIDataset
from duke import DUKEDataset
from lidc import LIDCDataset
from default import DEFAULTDataset
#from default import NCDataset
#from dataset import MRNetDataset, BRATSDataset, ADNIDataset, DUKEDataset, LIDCDataset, DEFAULTDataset, NCDataset
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Subset, ConcatDataset

# ============================================================================================

args = parser_def()
def get_dataset(cfg):
    private_train_dataset = DEFAULTDataset( args,
                                            mode = 'train',
                                            dataset = 'private')
    public_train_dataset = DEFAULTDataset(  args,
                                            mode = 'train',
                                            dataset = 'public')
    train_dataset = ConcatDataset([private_train_dataset, public_train_dataset])
    private_test_dataset = DEFAULTDataset(  args,
                                            mode = 'test',
                                            dataset = 'private')
    public_test_dataset = DEFAULTDataset(   args,
                                            mode = 'test',
                                            dataset = 'public')
    val_dataset = ConcatDataset([private_test_dataset, public_test_dataset])
    sampler = None
    return train_dataset, val_dataset, sampler
