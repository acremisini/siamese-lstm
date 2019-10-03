from loops import *
from utils.storage import *
import sys
import resource

# increase recursion limit to facilitate model saving
max_rec = 0x100000
resource.setrlimit(resource.RLIMIT_STACK, [0x100 * max_rec, resource.RLIM_INFINITY])
sys.setrecursionlimit(max_rec)

best_model = 'sig-avg_w-0.4_lr-0.9'

train = False
lr_search = False

torch.cuda.manual_seed(1)

if 'cuda' in glob.device:
    glob.use_gpu

''' Prep '''

# init data loader
dl = DataLoader()
loops = Loops()

if glob.pre_train:
    vocab, batch_dict = dl.load_data(corpus_names=['stspretrain_clean.p', 'ststrain_new.p', 'ststest.p', 'stsval_clean.p'])
else:
    vocab, batch_dict = dl.load_data(corpus_names=['ststrain_new.p', 'ststest.p', 'stsval_clean.p'])

pre_train_batches = batch_dict['pre_train'] if glob.pre_train else None
train_batches = batch_dict['train']
test_batches = batch_dict['test']
val_batches = batch_dict['val'] if 'val' in batch_dict else None

''' Train '''

if train:
    # initialize model
    model = SiameseLSTM(vocab).to(glob.device)
    print('Running computation on device:', glob.device)

    if lr_search:
        loops.lr_search(model=model,
                        batches=train_batches,
                        val_batches=val_batches,
                        end_lr=1.0,
                        num_iter=4000,
                        step_mode='exp',
                        log=False)

    model.train()
    epoch_loss = dict()

    # Pre-train

    if glob.pre_train:
        loss = loops.train_loop(batches=pre_train_batches,
                                val_batches=val_batches,
                                model=model,
                                num_epochs=glob.num_epochs_pre_train,
                                proc_name='Pre-Train')
        epoch_loss['pre_train'] = loss[0]
        epoch_loss['pre_train_val'] = loss[1]
        epoch_loss['pre_train_weighted_val'] = loss[2]

    # Train

    loss = loops.train_loop(batches=train_batches,
                            val_batches=val_batches,
                            model=model,
                            num_epochs=glob.num_epochs_train,
                            proc_name='Train')

    # Save
    print('\n')
    save_network(network=model,
                 network_label='{0}_w-{1}_lr-{2}'.format(glob.model_name,
                                                         glob.val_weight,
                                                         glob.learning_rate),
                 active_epoch='N',
                 save_directory=glob.models_dir)

    epoch_loss['train'] = loss[0]
    epoch_loss['train_val'] = loss[1]
    epoch_loss['train_weighted_val'] = loss[2]

    save_results(epoch_loss, '{0}_epoch-loss_w-{1}_lr-{2}.csv'.format(glob.model_name,
                                                                      glob.val_weight,
                                                                      glob.learning_rate))



else:
    model = SiameseLSTM(vocab).to(glob.device)
    load_network(network = model, path='models/{0}'.format(best_model))

''' Test '''
with torch.no_grad():

    # Test
    model.eval()
    model.lstm.eval()

    results = loops.zero_one_test_loop(batches=test_batches,
                                       model=model)

    # save_results(results, '{0}_performance_w-{1}_lr-{2}_0-1.csv'.format(glob.model_name,
    #                                                                     glob.val_weight,
    #                                                                     glob.learning_rate))


