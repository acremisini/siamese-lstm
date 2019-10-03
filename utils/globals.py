import os
import torch

class Globals():

    # PATHS
    home_dir = os.path.join(os.path.dirname(__file__), '..')
    data_dir = os.path.join(home_dir, 'data')
    results_dir = os.path.join(home_dir, 'results')
    models_dir = os.path.join(home_dir, 'models')
    plots_dir = os.path.join(home_dir, 'plots')

    # Model Parameters
    embedding_dim = 300
    hidden_dims = 50
    pre_train_batch_size = 32
    train_batch_size = 32
    test_batch_size = 4627 # 4627
    clip_value = 0.25
    learning_rate = 0.9#0.001 # 1e-2
    val_weight = 0.45
    max_lr = 0.005
    beta_1 = 0.1
    bi_dir = False
    model_name = 'l1'
    num_layers = 1
    dropout = 0.2 if num_layers > 1 else 0

    # Training
    overfit_data_size=None
    num_epochs_pre_train = 1000 #66 #40
    num_epochs_train = 1000 #375 #20
    report_freq = 40
    pre_train_patience = 8
    train_patience = 15

    # Switches
    expand_data = False
    pre_train = overfit_data_size is None

    # Testing
    run_data_integrity_tests = False
    run_embed_tests = False
    data_integrity_table = dict()

    # GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    device_count = 1 # torch.cuda.device_count() - not working
    use_gpu = lambda x=True: torch.set_default_tensor_type(torch.cuda.FloatTensor
                                                           if torch.cuda.is_available() and x
                                                           else torch.FloatTensor)


