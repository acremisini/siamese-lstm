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

    def lr_search(self, model, batches, val_batches, end_lr, num_iter, step_mode, log):
        lr_finder = LRFinder(model, model.optimizer, nn.MSELoss(), device="cuda")
        lr_finder.range_test(batches, val_loader=val_batches, end_lr=end_lr, num_iter=num_iter, step_mode=step_mode)
        lr_finder.plot(log_lr=False)

    def train_loop(self,batches,val_batches,model,num_epochs,proc_name):
        print('\n**********************************')
        print(proc_name.upper() + '\n')

        best_epoch = 0
        best_score = 1
        best_state_dict = dict()
        train_loss_list = []
        weighted_val_loss_list = []
        val_loss_list = []
        for epoch in range(num_epochs):
            # shuffle batches
            batches.init_epoch()

            epoch_loss = []
            i = 0
            for batch in batches:

                running_loss = []

                # pad data for this batch
                # padded = self.pad_batch(batch)
                # batch.s1 = padded[0]
                # batch.s2 = padded[1]

                batch.label = ((batch.label - 1) / 4.0)

                # empty gradients
                model.optimizer.zero_grad()

                # get prediction
                y_pred = model.forward(batch=batch)

                # get loss
                loss = model.get_loss(y_pred=y_pred, y=batch.label)
                epoch_loss.append(loss.data.item())
                running_loss.append(loss.data.item())

                # back-propagate loss
                loss.backward()

                # average gradients
                # model.avg_gradients()

                # step
                model.optimizer.step()

                if i % glob.report_freq == 0 and i != 0:
                    running_avg_loss = sum(running_loss) / len(running_loss)
                    print('%s | Epoch: %d | Training Batch: %d | Average loss since batch %d: %.16f' %
                          (proc_name,epoch, i, i - glob.report_freq, running_avg_loss))
                i += 1

                if hasattr(model, 'scheduler'):
                    if 'CyclicLR' in str(type(model.scheduler)):
                        model.scheduler.batch_step()
            # epoch_i done, process:

            avg_training_loss = sum(epoch_loss) / len(epoch_loss)
            train_loss_list.append((epoch, avg_training_loss))
            print('\nAverage %s batch loss at epoch %d: %.16f \n' % (proc_name, epoch, avg_training_loss))

            # lr update and improvement checking
            print('======')
            model.eval()
            val_loss = self.eval_loop(batches=val_batches, epoch_num=epoch, model=model)
            model.train()

            weighted_val = (val_loss + avg_training_loss*glob.val_weight) / 2.0
            val_loss_list.append((epoch,val_loss))

            if weighted_val < best_score:
                best_score = weighted_val
                best_epoch = epoch
                best_state_dict = model.state_dict()
            weighted_val_loss_list.append((epoch, weighted_val))
            print('\nBest epoch was', abs(best_epoch - epoch),'epochs ago','({0})'.format(best_score))

            patience = glob.pre_train_patience if 'pre' in proc_name.lower() else glob.train_patience
            if abs(best_epoch - epoch) > patience:
                print('Exiting early due to lack of progress, best validated epoch is {0}, with score {1}'.format(best_epoch, best_score))
                model.load_state_dict(best_state_dict)
                break
            print('======\n')

        print('\n-----> %s procedure concluded after %d epochs total. Best validated epoch: %d.'
              % (proc_name, epoch, best_epoch))
        return [train_loss_list,val_loss_list,weighted_val_loss_list]

    def eval_loop(self,batches,epoch_num,model):

        print('EVAL')

        total_valid_loss = list()

        for batch in iter(batches):
            batch.label = ((batch.label - 1) / 4.0)

            y_pred = model.forward(batch=batch)
            loss = model.get_loss(y_pred, batch.label)
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
    def make_reg_dataset(self, batches, model, reg_model=None):

        print('\n**********************************')
        print('CONSTRUCTING REGRESSION DATASET\n')

        y_hat = []
        y = []

        for batch in iter(batches):
            y_hat += model.forward(batch=batch).tolist()
            y += batch.label.tolist()

        # calculate results (for reference)
        X = np.array(list(range(len(y_hat))))
        y_scaled = [(y_i -1)/4.0 for y_i in y]
        pearson = meas.pearsonr(y_hat,y_scaled)[0]
        spearman = meas.spearmanr(y_hat,y_scaled)[0]
        mse = np.mean(np.square(np.subtract(y_hat,y_scaled)))

        # report results
        print('===================================================\n')
        print('''Testing procedure concluded, overfit results are {0}:\n
              Pearson: {1},\n
              Spearman: {2},\n
              MSE: {3},
              '''.format('[0,1]',pearson,spearman,mse))
        print('===================================================')

        #plot
        plt.xlabel('Test Sample')
        plt.ylabel('Similarity')
        plt.scatter(x=X,y=y_scaled,
                    color='red',
                    marker = "o",
                    s=0.3,
                    label='Labels')
        plt.scatter(x=X,y=y_hat,
                    color='blue',
                    s=0.5,
                    marker = 0,
                    label='Predictions')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(glob.plots_dir,'{0}_plot_w-{1}_lr-{2}_{3}.pdf').format(glob.model_name,
                                                                                        glob.val_weight,
                                                                                        glob.learning_rate,
                                                                                        '0-1'))

        plt.close()

        y_hat = np.array(y_hat).reshape(-1,1)
        y = np.array(y).reshape(len(y),)

        return [[y_hat,y], {'pearson': pearson,
                            'spearman':spearman,
                            'mse':mse}]

    def reg_test_loop(self, batches, model, reg_model):

        print('\n**********************************')
        print('TEST\n')

        y_hat = []
        y = []

        # get (y_hat, y)
        for batch in iter(batches):
            y_hat += model.forward(batch=batch).tolist()
            y += batch.label.tolist()

        # calculate performance
        y_hat = np.clip(reg_model.predict(np.array(y_hat).reshape(-1,1)),a_min=1.0,a_max=5.0)
        X = np.array(list(range(len(y_hat))))
        pearson = meas.pearsonr(y_hat,y)[0]
        spearman = meas.spearmanr(y_hat, y)[0]
        mse = np.mean(np.square(np.subtract(y_hat, y)))


        # report results
        print('===================================================\n')
        print('''Testing procedure concluded, results are {0}:\n
              Pearson: {1},\n
              Spearman: {2},\n
              MSE: {3},
              '''.format('[1,5]',pearson,spearman,mse))
        print('===================================================')

        #plot
        X = np.array(X)
        plt.xlabel('Test Sample')
        plt.ylabel('Similarity')
        plt.scatter(x=X,y=y,
                    color='red',
                    marker = "o",
                    s=0.3,
                    label='Labels')
        plt.scatter(x=X,y=y_hat,
                    color='blue',
                    s=0.5,
                    marker = 0,
                    label='Predictions')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(glob.plots_dir,'{0}_plot_w-{1}_lr-{2}_{3}.pdf').format(glob.model_name,
                                                                                        glob.val_weight,
                                                                                        glob.learning_rate,
                                                                                        '1-5'))

        plt.close()
        # [y_hat.reshape(-1, 1), y.reshape((len(y),))]

        return {'pearson': pearson,
                'spearman':spearman,
                'mse':mse}


    def pad_batch(self, batch):
        max_len = max(f.size(0) for f in [batch.s1, batch.s2])
        padded = []
        for sent in [batch.s1, batch.s2]:
            if sent.size(0) < max_len:
                pad_size = [max_len - sent.size(0), sent.size(1)]
                pad = torch.ones(pad_size).long().to(glob.device)
                sent = torch.cat((sent, pad))
            padded.append(sent)
        return padded
