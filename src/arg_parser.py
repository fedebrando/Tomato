
import argparse

def check_prob(val):
    f = float(val)
    if f < 0 or f > 1:
      raise argparse.ArgumentTypeError('Error: %s is not a probability' % val)
      
    return f

def get_train_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--bs', type=int, default=16, help='number of elements in batch size')
    parser.add_argument('--print_every', type=int, default=5, help='print losses every N iteration')
    parser.add_argument('--early_stop', type=int, default=5, help='number of non improvements on validation accuracy to stpo training')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--opt', type=str, default='Adam', choices=['SGD', 'Adam'], help='optimizer used for training')
    
    parser.add_argument('--aug_flip', type=check_prob, default=0, help='data augmentation with flip')
    parser.add_argument('--aug_rotate', type=check_prob, default=0, help='data augmentation with rotate')
    parser.add_argument('--aug_jitter', type=check_prob, default=0, help='data augmentation with color jitter')
    parser.add_argument('--aug_crop', type=check_prob, default=0, help='data augmentation with crop')
    
    parser.add_argument('--refine_model', type=str, default='', help='refine model of path received')
    
    return parser.parse_args()

def get_test_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='best', help='model name to be tested')
    
    return parser.parse_args()
