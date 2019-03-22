from loops import *
from utils.storage import *
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import torch.nn as nn
import pickle

torch.cuda.manual_seed(5)

if 'cuda' in glob.device:
    glob.use_gpu

if glob.expand_data:
    ex = STSExtender(selectivity_rate=0.4)
    ex.extend_data('ststrain.p')

dl = DataLoader()
loops = Loops()

# bucket and batch data
if glob.pre_train:
    vocab, batch_list = dl.load_data(corpus_names=['stspretrain.p','ststrain_aug2.p'],is_train=True)
else:
    vocab, batch_list = dl.load_data(corpus_names=['ststrain.p'], is_train=True, debug=glob.debug)


# initialize model
model = SiameseLSTM(vocab).to(glob.device)
model.train()
print('Running computation on device:',glob.device)

# Pre-train

if glob.pre_train:
    batches = batch_list[0]

    if glob.device_count > 1:
        print('Running on', glob.device_count, 'GPUs')
        model = nn.DataParallel(model)

    loops.train_loop(batches=batches,model=model,num_epochs=glob.num_epochs_pre_train,proc_name='Pre-Train',early_stopping=False, lr_search=False)

# Train

batches = batch_list[1] if glob.pre_train else batch_list[0]
loops.train_loop(batches=batches,model=model,num_epochs=glob.num_epochs_train,proc_name='Train',early_stopping=False, lr_search=False)

# Train a regression model to take [0,1] predictions to [1,5]

if glob.debug is None:
    _,batch_list = dl.load_data(corpus_names=['ststrain.p'], is_train=False)
    batches = batch_list[0]
else:
    _, batch_list = dl.load_data(corpus_names=['ststrain.p'], is_train=False,debug=glob.debug)
    batches = batch_list[0]
model.eval()
y_pred, y = loops.test_loop(batches=batches,model=model)


sim_svr = GridSearchCV(SVR(kernel='rbf', gamma=3.1), cv=5, param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)})
print('Fitting regression model...')
sim_svr.fit(X=y_pred, y=y)
print('Done, saving')
svr_path = os.path.join(glob.models_dir, 'sim_svr.p')
with open(svr_path, 'wb') as f:
    pickle.dump(sim_svr, f)
print('Trained SVR model saved to %s' % svr_path)


# Test

if glob.debug is None:
    _, batch_list = dl.load_data(corpus_names=['ststest.p'],is_train=False)
    batches = batch_list[0]
else:
    _, batch_list = dl.load_data(corpus_names=['ststest.p'], is_train=False, debug=10)
    batches = batch_list[0]

loops.test_loop(batches=batches, model=model, reg_model=sim_svr)

# Save final model
print('\n')
save_network(network=model,network_label='final_model',active_epoch='N',save_directory=glob.models_dir)
