
import argparse

def get_train_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--bs', type=int, default=16, help='number of elements in batch size')
    parser.add_argument('--print_every', type=int, default=5, help='print losses every N iteration')
    parser.add_argument('--early_stop', type=int, default=5, help='number of non improvements on validation accuracy')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--opt', type=str, default='Adam', choices=['SGD', 'Adam'], help = 'optimizer used for training')
    
    return parser.parse_args()

def get_test_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='best', help = 'model name to be tested')
    
    return parser.parse_args()
