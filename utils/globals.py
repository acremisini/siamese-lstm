import os
import torch

class Globals():

    # PATHS
    home_dir = os.path.join(os.path.dirname(__file__), '..')
    data_dir = os.path.join(home_dir, 'data')
    models_dir = os.path.join(home_dir, 'models')
    plots_dir = os.path.join(home_dir, 'plots')

    # Model Parameters
    embedding_dim = 300
    hidden_dims = 50
    num_layers = 1
    pre_train_batch_size = 32
    train_batch_size = 1
    test_batch_size = 1
    clip_value = 0.25
    learning_rate = .0025 # 1e-2
    beta_1 = 0.1
    min_lr = 1
    max_lr = 2
    bi_dir = False

    # Training
    debug=None
    num_epochs_pre_train = 1 #66 #40
    num_epochs_train = 1 #375 #20
    report_freq = 64
    # arbitrary?
    start_early_stopping_pre = num_epochs_pre_train*0.60
    patience_pre = num_epochs_pre_train*0.20
    start_annealing_pre = num_epochs_pre_train*0.60
    annealing_factor = 0.75

    start_early_stopping = num_epochs_train*0.60
    patience = 10
    start_annealing = num_epochs_train*0.60

    # Switches
    expand_data = False
    pre_train = True

    # GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    device_count = 1 # torch.cuda.device_count() - not working
    use_gpu = lambda x=True: torch.set_default_tensor_type(torch.cuda.FloatTensor
                                                           if torch.cuda.is_available() and x
                                                           else torch.FloatTensor)


