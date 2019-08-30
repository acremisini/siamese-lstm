from loops import *
from utils.storage import *
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import pickle
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

# expand data
if glob.expand_data:
    ex = STSExtender(selectivity_rate=0.4)
    ex.extend_data('ststrain.p')

# init data loader
dl = DataLoader()
loops = Loops()

# bucket and batch data
# must be pretrain_clean and val_clean if validate (val_clean is subset of pretrain)
# otherwise just pretrain
if glob.pre_train:
    vocab, batch_dict = dl.load_data(corpus_names=['stspretrain_clean.p', 'ststrain_new.p', 'ststest.p', 'stsval_clean.p'])
else:
    vocab, batch_dict = dl.load_data(corpus_names=['ststrain_new.p', 'ststest.p', 'stsval_clean.p'])

pre_train_batches = batch_dict['pre_train'] if glob.pre_train else None
train_batches = batch_dict['train']
test_batches = batch_dict['test']
val_batches = batch_dict['val'] if 'val' in batch_dict else None


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

    ''' Train and Test '''

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
    epoch_loss['train'] = loss[0]
    epoch_loss['train_val'] = loss[1]
    epoch_loss['train_weighted_val'] = loss[2]

    save_results(epoch_loss, '{0}_epoch-loss_w-{1}_lr-{2}.csv'.format(glob.model_name,
                                                                      glob.val_weight,
                                                                      glob.learning_rate))

else:
    model = SiameseLSTM(vocab).to(glob.device)
    load_network(network = model, path='models/{0}'.format(best_model))

skip = True

if not skip:
    with torch.no_grad():

        # make regression dataset
        results = loops.make_reg_dataset(batches=train_batches, model=model)
        y_pred, y = results[0][0], results[0][1]

        save_results(results[1], '{0}_performance_w-{1}_lr-{2}_0-1.csv'.format(glob.model_name,
                                                                               glob.val_weight,
                                                                               glob.learning_rate))
        # fit regression model

        load_reg = False
        svr_path = os.path.join(glob.models_dir, '{0}_sim-svr_w-{1}_lr-{2}'.format(glob.model_name,
                                                                                   glob.val_weight,
                                                                                   glob.learning_rate))

        if load_reg:
            print('Loading regression model')
            sim_svr = pickle.load(open(svr_path, 'rb'), encoding='utf-8')
            print('..... Done')
        else:
            sim_svr = GridSearchCV(SVR(kernel='rbf', gamma=3.1), cv = 5,param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)})
            print('Fitting regression model...')
            sim_svr.fit(X=y_pred, y=y)
            print('Done, saving')

            with open(svr_path, 'wb') as f:
                pickle.dump(sim_svr, f)
            print('Trained SVR model saved to %s' % svr_path)


        # Test
        results = loops.reg_test_loop(batches=test_batches,
                                      model=model,
                                      reg_model=sim_svr)

        save_results(results, '{0}_performance_w-{1}_lr-{2}_1-5.csv'.format(glob.model_name,
                                                                            glob.val_weight,
                                                                            glob.learning_rate))
        # Save
        print('\n')
        save_network(network=model,
                     network_label='{0}_w-{1}_lr-{2}'.format(glob.model_name,
                                                             glob.val_weight,
                                                             glob.learning_rate),
                     active_epoch='N',
                     save_directory=glob.models_dir)
else:
    with torch.no_grad():
        results = loops.zero_one_test_loop(batches=test_batches,
                                           model=model)
