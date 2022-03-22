import torch
import os


def get_rl_dataset(path, dataset,
                   quality='medium', type='1', noise='0.5'):
    fn = os.path.join(path, dataset, f'{quality}-type_{type}-noise_{noise}.pt')
    ckpt = torch.load(fn)
    states, actions, next_states = ckpt['states'], ckpt['actions'], ckpt['next_states']
    X = torch.cat((states, actions), dim=1).cpu().numpy()
    y = next_states.cpu().numpy()
    return X, y

def get_epochs(dataset):
    if dataset in ['meps_19', 'meps_20', 'meps_21']:
        return 800
    elif dataset in ['facebook_1', 'facebook_2', 'blog_data']:
        return 1500
    elif dataset == 'kin8nm':
        return 6000
    elif dataset == 'naval':
        return 350
    elif dataset == 'bio':
        return 2500


def lr_power_decay(epoch, decay_factor, decay_epochs):
    # assumes decay_epochs has 2 elements in increasing order
    if epoch < decay_epochs[0]:
        return 1
    elif epoch < decay_epochs[1]:
        return decay_factor
    else:
        return (decay_factor ** 2)