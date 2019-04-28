import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import scipy.stats as meas
from utils.Adadelta import Adadelta as custom_Adadelta
from utils.cyclic_lr import CyclicLR
from utils.parameter_initialization import *
from utils.globals import Globals as glob
from unit_tests.test_embeds import TestEmbeds
from unit_tests.test_data import TestData

if 'cuda' in glob.device:
    glob.use_gpu

class LSTM(nn.Module):

    def __init__(self, vocab):
        super(LSTM, self).__init__()
        self.embed_test = TestEmbeds()

        self.vocab = vocab
        self.vocab_size = vocab.vectors.size(0)
        self.embedding_dim = vocab.vectors.size(1)
        self.name = 'lstm_unit'

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=glob.hidden_dims,
                            batch_first=False,
                            num_layers=glob.num_layers,
                            dropout = glob.dropout)

        self.initialize_parameters()

    def initialize_parameters(self):
        """ Initializes network parameters. """
        state_dict = self.lstm.state_dict()
        for key in state_dict.keys():
            if 'weight' in key:
                hidden_dim = state_dict[key].size(0) / 4
                embed_dim = state_dict[key].size(1)
                # W
                if 'ih' in key:
                    state_dict[key] = Variable(torch.nn.init.normal_(state_dict[key],mean=0,std=.2) / torch.sqrt(torch.from_numpy(np.array(hidden_dim*embed_dim))))
                # U
                elif 'hh' in key:
                    state_dict[key] = Variable(torch.nn.init.normal_(state_dict[key], mean=0, std=.2) / torch.sqrt(torch.from_numpy(np.array(hidden_dim))))
            # b
            if 'bias' in key:
                hidden_dim = Variable((torch.tensor(state_dict[key].size(0) / 4).long()))
                # from paper
                state_dict[key] = Variable(torch.nn.init.uniform_(state_dict[key], a=-0.5,b=0.5))
                # paper says 2.5, their code has 1.5
                state_dict[key][hidden_dim:hidden_dim*2] = Variable(torch.tensor([2.5]))

        self.lstm.load_state_dict(state_dict)

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)

        h = torch.zeros(glob.num_layers,batch_size,glob.hidden_dims).to(glob.device)
        h = Variable(torch.nn.init.normal_(h, mean=0, std=.01).to(glob.device))
        #h = Variable(torch.nn.init.xavier_normal_(h).to(glob.device))

        c = torch.zeros(glob.num_layers,batch_size,glob.hidden_dims).to(glob.device)
        c = Variable(torch.nn.init.normal_(c, mean=0, std=.01).to(glob.device))
        #c = Variable(torch.nn.init.xavier_normal_(c).to(glob.device))

        # force model to learn initial states
        # init_hidden = nn.Parameter(torch.randn(glob.num_layers,batch_size,glob.hidden_dims).type(torch.FloatTensor), requires_grad=True).to(glob.device)
        # init_cell = nn.Parameter(torch.randn(glob.num_layers, batch_size, glob.hidden_dims).type(torch.FloatTensor),requires_grad=True).to(glob.device)

        return h,c

    def forward(self, words_idx, batch_size, hidden, cell):
        """ batch_size words are fed in parallel to the network, with shape (1xbatch_size).
        After embedding lookup, the output has shape (1, batch_size, embedding_dim) """

        embeds = self.vocab.vectors[words_idx].to(glob.device)

        if glob.run_embed_tests:
            self.embed_test.test_embedLookup(embeds=embeds,
                                             words_idx=words_idx,
                                             vocab=self.vocab)

        embeds = embeds.view(1, batch_size, self.embedding_dim)


        output, (hidden,cell) = self.lstm(embeds, (hidden,cell))

        return output, hidden, cell



class SiameseLSTM(nn.Module):

    def __init__(self, vocab):
        super(SiameseLSTM, self).__init__()
        self.embed_test = TestEmbeds()
        self.vocab = vocab

        # Initialize one lstm, this will by default share variables across inputs
        self.lstm = LSTM(self.vocab)
        # self.lstm_b = LSTM(self.vocab)

        # self.lstm_b.load_state_dict(self.lstm_a.state_dict())

        # Define loss function
        self.loss_function = nn.MSELoss()

        self.optimizer = optim.Adadelta(params=self.lstm.parameters(),
                                        lr=glob.learning_rate,
                                        rho=0.95,
                                        eps=1e-6)
        # self.scheduler = CyclicLR(self.optimizer,
        #                           base_lr=glob.learning_rate,
        #                           max_lr=glob.max_lr,
        #                           step_size=60000,
        #                           mode='triangular2')


    def forward(self, batch):
        """ Perform forward pass through the network. """

        if glob.run_embed_tests:
            self.embed_test.test_batchKeys(batch=batch,
                                           vocab=self.vocab,
                                           gold_dict=glob.data_integrity_table,
                                           proc_name='batch key in lstm')
            self.embed_test.test_batchLabels(batch=batch,
                                             vocab=self.vocab,
                                             gold_dict=glob.data_integrity_table,
                                             proc_name='batch label in lstm')

        # batches are of shape (sentence, num_batches)

        # run this batch through lstm_a
        hidden_a_t, cell_a_t = self.lstm.init_hidden(batch.s1.size(1))
        for t_i in range(batch.s1.size(0)):
            out_a, hidden_a_t, cell_a_t = self.lstm(words_idx=batch.s1[t_i, :],
                                                    batch_size = batch.s1.size(1),
                                                    hidden = hidden_a_t,
                                                    cell = cell_a_t)

        # run this batch through lstm_b
        hidden_b_t, cell_b_t = self.lstm.init_hidden(batch.s2.size(1))
        for t_i in range(batch.s2.size(0)):
            out_b, hidden_b_t, cell_b_t = self.lstm(words_idx=batch.s2[t_i, :],
                                                    batch_size=batch.s2.size(1),
                                                    hidden=hidden_b_t,
                                                    cell=cell_b_t)

        # Get similarity predictions:
        if glob.num_layers > 1:
            dif = hidden_a_t[-1].squeeze() - hidden_b_t[-1].squeeze()
        else:
            dif = hidden_a_t.squeeze() - hidden_b_t.squeeze()

        norm = torch.norm(dif,
                          p=1,
                          dim=1 if dif.dim() > 1 else 0)
        y_hat = torch.exp(-norm)
        y_hat = torch.clamp(y_hat, min=1e-7, max=1.0 - 1e-7)

        return torch.reshape(y_hat, (-1,))



    def get_loss(self, y_pred, y):
        ''' Compute MSE between predictions and scaled gold labels '''
        return self.loss_function(y_pred,y)

    def avg_gradients(self):
        grads = dict()
        for name, param in self.named_parameters():
            if param.requires_grad:
                grads[name] = param.grad.data

        avg_grads = dict()
        params = ['lstm.weight_ih_l0', 'lstm.weight_hh_l0', 'lstm.bias_ih_l0', 'lstm.bias_hh_l0']

        # record update for each parameter, averaging gradients from lstm_a and lstm_b
        # get gradients for lstm_a
        for p in params:
            avg_grads[p.replace('lstm.', '')] = grads['lstm_a.' + p]
        # add gradients from lstm_b and average
        for p in params:
            avg_grads[p.replace('lstm.', '')] += grads['lstm_b.' + p]
            avg_grads[p.replace('lstm.', '')] = avg_grads[p.replace('lstm.', '')] / 2.0

        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'lstm_a' in name:
                    param.grad.data = avg_grads[name.replace('lstm_a.lstm.', '')]
                elif 'lstm_b' in name:
                    param.grad.data = avg_grads[name.replace('lstm_b.lstm.', '')]

# preds = [y.clone().cpu().detach().item() for y in y_pred]
# gold = [i.clone().cpu().detach().item() for i in y]
# pearson = meas.pearsonr(preds, gold)[0]
# if pearson > 0.8:
#     print(pearson)
#     res = list(zip(preds,gold))
#     for r in res:
#         print(r[0], r[1])
#     print('-----')