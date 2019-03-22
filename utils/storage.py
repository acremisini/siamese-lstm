import os
import torch


def save_network(network, network_label, active_epoch, save_directory):
    """ Saves the parameters of the specified network under the specified path. """
    file_name = '%s_%s' % (str(active_epoch), network_label)
    save_path = os.path.join(save_directory, file_name)
    torch.save(network.cpu().state_dict(), save_path)
    print('Network %s saved following the completion of epoch %s | Location: %s' %
          (network_label, str(active_epoch), save_path))


def load_network(network, network_label, target_epoch, load_directory):
    """ Helper function for loading network work. """
    load_filename = '%s_%s' % (str(target_epoch), network_label)
    load_path = os.path.join(load_directory, load_filename)
    network.load_state_dict(torch.load(load_path))
    print('Network %s, version %s loaded from location %s' % (network_label, target_epoch, load_path))