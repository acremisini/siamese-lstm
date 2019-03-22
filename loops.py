from siamese_lstm.lstm import *
from utils.data_extender import STSExtender
from utils.data_loader import DataLoader
from utils.globals import Globals as glob
from utils.globals import *
import numpy as np
import scipy.stats as meas
import matplotlib.pyplot as plt
import math
from utils.lr_finder import LRFinder

from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.nn as nn

class Loops():

    def __init__(self):
        self.best_validation_accuracy = 0
        self.epochs_without_improvement = 0
        self.total_loss = []

    def train_loop(self,batches,model,num_epochs,proc_name,early_stopping,lr_search=False):
        print('\n**********************************')
        print(proc_name.upper() + '\n')

        # load validation set
        if glob.debug is None:
            dl = DataLoader()
            _, val_batch_list = dl.load_data(corpus_names=['stsval.p'],is_train=False)
            val_batches = val_batch_list[0]

        # learning rate search
        if lr_search:
            lr_finder = LRFinder(model, model.optimizer, nn.MSELoss(), device="cuda")
            lr_finder.range_test(batches,val_loader=val_batches if glob.debug is None else None,end_lr=1e-7, num_iter=100, step_mode="exp")
            #lr_finder.range_test(batches,val_loader=val_batches,end_lr=100, num_iter=100)
            lr_finder.plot(log_lr=True)

        best_epoch = 0
        best_score = 1
        for epoch in range(num_epochs):

            epoch_loss = []
            i = 0
            for batch in iter(batches):
                running_loss = []

                # get data for this batch
                batch.s1 = batch.s1.to(glob.device)
                batch.s2 = batch.s2.to(glob.device)
                batch.label = batch.label.to(glob.device)

                # empty gradients
                model.optimizer.zero_grad()

                # get prediction
                y_pred = model.forward(batch=batch, is_singleton=False if len(batch.label) > 1 else True)

                # get loss
                # is_test = False, so batch.label will be scaled to [0,1]
                loss = model.get_loss(y_pred=y_pred, y=batch.label, is_test=False)
                epoch_loss.append(loss.data.item())
                running_loss.append(loss.data.item())

                # back-propagate loss
                loss.backward()

                # step
                model.optimizer.step()

                if i % glob.report_freq == 0 and i != 0:
                    running_avg_loss = sum(running_loss) / len(running_loss)
                    print('%s | Epoch: %d | Training Batch: %d | Average loss since batch %d: %.16f' %
                          (proc_name,epoch, i, i - glob.report_freq, running_avg_loss))
                i += 1

            avg_training_accuracy = sum(epoch_loss) / len(epoch_loss)
            print('\nAverage %s batch loss at epoch %d: %.16f \n' % (proc_name, epoch, avg_training_accuracy))

            # lr update and improvement checking
            print('======')
            if glob.debug is None:

                model.eval()
                val_loss = self.eval_loop(batches=val_batches, epoch_num=epoch, model=model)
                model.train()
            else:
                val_loss = avg_training_accuracy

            if 'CyclicLR' in str(type(model.scheduler)):
                model.scheduler.batch_step()
            elif 'Adadelta' not in str(type(model.optimizer)):
                model.scheduler.step(val_loss)
            if val_loss < best_score:
                best_score = val_loss
                best_epoch = epoch
            print('\nBest epoch was', abs(best_epoch - epoch),'epochs ago','({0})'.format(best_score))
            if abs(best_epoch - epoch) > glob.patience:
                print('Exiting early due to lack of progress, best validated epoch is {0}, with score {1}'.format(best_epoch, best_score))
                break
            print('======\n')

            # epoch_i done

        print('\n-----> %s procedure concluded after %d epochs total. Best validated epoch: %d.'
              % (proc_name, epoch, best_epoch))

    def eval_loop(self,batches,epoch_num,model):
        # print('\n**********************************')
        print('EVAL')
        # print('Running eval loop...')
        total_valid_loss = list()

        for batch in iter(batches):
            batch.s1 = batch.s1.to(glob.device)
            batch.s2 = batch.s2.to(glob.device)
            batch.label = batch.label.to(glob.device)
            # have to rescale here
            batch.label = (batch.label - 1) / 4.0

            y_pred = model.forward(batch=batch, is_singleton=True)
            loss = model.get_loss(y_pred, batch.label, is_test=True)
            total_valid_loss.append(loss.item())


        # Report fold statistics

        mse = sum(total_valid_loss) / len(total_valid_loss)*1.0
        print('\nAverage validation fold accuracy at epoch %d: %.16f' % (epoch_num, mse))
        # print('\n**********************************')

        return mse

    # during training, the labels are scaled to [0,1] (this happens inside the model)
    # during testing, the predictions are scaled to [1,5] using a regression model
    # this loop is used without a reg_model to get predictions in [0,1] and use this to train a
    # regression model
    # with a reg_model passed in, we must use it to send [0,1] predictions given by model
    # to [1,5]
    def test_loop(self,batches,model,reg_model=None):
        preds = []
        y = []
        y_plot = []
        pred_plot = []

        print('\n**********************************')
        if reg_model is None:
            print('REGRESSION\n')
        else:
            print('TEST\n')
        i = 1
        X = []
        for batch in iter(batches):
            # get data for sentences and label
            batch.s1 = batch.s1.to(glob.device)
            batch.s2 = batch.s2.to(glob.device)
            batch.label = batch.label.to(glob.device)

            # make prediction
            y_pred = model.forward(batch=batch, is_singleton=True)

            if reg_model:
                # use model to scale model's [0,1] prediction to [1,5]
                y_pred = np.clip(reg_model.predict(y_pred.cpu().detach().numpy().reshape(1, -1)), a_min=1, a_max=5)

            preds.append(y_pred)
            y.append(batch.label)

            X.append(i)
            i += 1

        if reg_model is None:
            preds_np = np.array([float(p_i.cpu().detach().numpy()) for p_i in preds])
        else:
            preds_np = np.array([p_i[0] for p_i in preds])

        y_np = np.array([float(y_i.cpu().detach().numpy()) for y_i in y])

        # calculate results
        pearson = meas.pearsonr(preds_np,y_np)[0]
        spearman = meas.spearmanr(y_np, preds_np)[0]
        if reg_model is None:
            y_np_scale = [(y_i-1)/4 for y_i in y_np]
            mse = np.mean(np.square(y_np_scale - preds_np))
        else:
            mse = np.mean(np.square(y_np - preds_np))

        # report results
        print('===================================================\n')
        print('''Testing procedure concluded, final results are {0}:\n
              Pearson: {1},\n
              Spearman: {2},\n
              MSE: {3},
              '''.format('[0,1]' if reg_model is None else '[1,5]',pearson,spearman,mse))
        print('===================================================')

        #plot
        X = np.array(X)
        plt.xlabel('test sample')
        plt.ylabel('similarity')
        plt.scatter(x=X,y=y_np if reg_model else y_np_scale,color='red',s=0.1)
        plt.scatter(x=X,y=preds_np,color='blue',s=0.05)
        plt.savefig(os.path.join(glob.plots_dir,'results_' + str(reg_model)[0:4] + '.png'))
        plt.close()

        return preds_np.reshape(-1,1),y_np.reshape((len(y_np),))
