import os
import torch
from utils.globals import Globals as glob


def save_network(network, network_label, active_epoch, save_directory):
    """ Saves the parameters of the specified network under the specified path. """
    file_name = network_label
    save_path = os.path.join(save_directory, file_name)
    torch.save(network.cpu().state_dict(), save_path)
    print('Network %s saved following the completion of epoch %s | Location: %s' %
          (network_label, str(active_epoch), save_path))


def load_network(network, path):
    """ Helper function for loading network work. """
    network.load_state_dict(torch.load(path))
    print('Network loaded from location %s' % (path))

def save_results(result, path):
    # y_pred, y
    if isinstance(result, tuple):
        out = 'y_hat, y\n'
        for t in zip(result[0], result[1]):
            y_hat = t[0][0]
            y = t[1]
            out += '{0},{1}\n'.format(y_hat, y)
        with open(glob.results_dir + "/" + path, "w") as text_file:
            print("{}".format(out), file=text_file)
    #
    elif isinstance(result, dict):
        if 'train' in result:
            out = 'epoch,avg_loss\n'
            for t in result['train']:
                out += '{0},{1}\n'.format(t[0], t[1])
            with open(glob.results_dir + "/" + path.replace(glob.model_name,glob.model_name + '_train'), "w") as text_file:
                print("{}".format(out), file=text_file)

        if 'pre_train' in result:
            out = 'epoch,avg_loss\n'
            for t in result['pre_train']:
                out += '{0},{1}\n'.format(t[0], t[1])
            with open(glob.results_dir + "/" + path.replace(glob.model_name,glob.model_name + '_pre-train'), "w") as text_file:
                print("{}".format(out), file=text_file)

        if 'train_val' in result:
            out = 'epoch,val_loss\n'
            for t in result['train_val']:
                out += '{0},{1}\n'.format(t[0], t[1])
            with open(glob.results_dir + "/" + path.replace(glob.model_name,glob.model_name + '_train-val'), "w") as text_file:
                print("{}".format(out), file=text_file)

        if 'pre_train_val' in result:
            out = 'epoch,val_loss\n'
            for t in result['pre_train_val']:
                out += '{0},{1}\n'.format(t[0], t[1])
            with open(glob.results_dir + "/" + path.replace(glob.model_name,glob.model_name + '_pre-train-val'), "w") as text_file:
                print("{}".format(out), file=text_file)

        if 'train_weighted_val' in result:
            out = 'epoch,weighted_val_loss\n'
            for t in result['train_weighted_val']:
                out += '{0},{1}\n'.format(t[0], t[1])
            with open(glob.results_dir + "/" + path.replace(glob.model_name, glob.model_name + '_train-weighted-val'),
                      "w") as text_file:
                print("{}".format(out), file=text_file)

        if 'pre_train_weighted_val' in result:
            out = 'epoch,weighted_val_loss\n'
            for t in result['pre_train_weighted_val']:
                out += '{0},{1}\n'.format(t[0], t[1])
            with open(glob.results_dir + "/" + path.replace(glob.model_name, glob.model_name + '_pre-train-weighted-val'),
                      "w") as text_file:
                print("{}".format(out), file=text_file)

        if 'pearson' in result:
            out = 'pearson,spearman,mse\n'
            out += '{0},{1},{2}'.format(result['pearson'],
                                        result['spearman'],
                                        result['mse'])
            with open(glob.results_dir + "/" + path, "w") as text_file:
                print("{}".format(out), file=text_file)