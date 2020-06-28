import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Dictionary, VQAFeatureDataset
import base_model
from train import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # here is the random seed to ensure the reproducibility of experimental results
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    #Load train and test set
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', dictionary)
    eval_dset = VQAFeatureDataset('val', dictionary)
    train_loader = DataLoader(train_dset, args.batch_size, shuffle=True, num_workers=1)
    eval_loader = DataLoader(eval_dset, args.batch_size, shuffle=True, num_workers=1)

    #Construct Model
    model = base_model.build(train_dset, args.num_hid).cuda()
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    model = nn.DataParallel(model).cuda()

    #train model
    train(model, train_loader, eval_loader, args.epochs, args.output)
