import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.cyclic_lr import CyclicLR

from utils.parameter_initialization import *
from utils.globals import Globals as glob

if 'cuda' in glob.device:
    glob.use_gpu

class LSTM(nn.Module):

    def __init__(self, vocab):
        super(LSTM, self).__init__()

        self.vocab_size = vocab.vectors.size(0)
        self.embedding_dim = vocab.vectors.size(1)
        self.name = 'lstm_unit'

        self.embedding_table = nn.Embedding.from_pretrained(vocab.vectors, freeze=True)


        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=glob.hidden_dims,
                            num_layers=glob.num_layers,
                            batch_first=True if glob.num_layers > 1 else False,
                            dropout=0,
                            bidirectional=glob.bi_dir)

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
                    state_dict[key] = Variable(torch.nn.init.normal_(state_dict[key],mean=0,std=0.2) / torch.sqrt(torch.from_numpy(np.array(hidden_dim*embed_dim))))
                # U
                elif 'hh' in key:
                    state_dict[key] = Variable(torch.nn.init.normal_(state_dict[key], mean=0, std=0.2) / torch.sqrt(torch.from_numpy(np.array(hidden_dim))))
            # b
            if 'bias' in key:
                hidden_dim = Variable((torch.tensor(state_dict[key].size(0) / 4).long()))
                # from paper
                state_dict[key] = Variable(torch.nn.init.uniform_(state_dict[key], a=-0.5,b=0.5))
                # paper says 2.5, their code has 1.5
                state_dict[key][hidden_dim:hidden_dim*2] = Variable(torch.tensor([2.5]))

        self.lstm.load_state_dict(state_dict)

    def init_hidden(self, batch_size):
        bi = 2 if glob.bi_dir else 1
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # randn samples from a N(0,1) distribution

        # init_hidden = Variable(torch.randn(glob.num_layers,batch_size,glob.hidden_dims)).to(glob.device)
        # init_cell = Variable(torch.randn(glob.num_layers,batch_size,glob.hidden_dims)).to(glob.device)

        h = torch.zeros(glob.num_layers*bi,batch_size,glob.hidden_dims)
        # init_hidden = Variable(torch.nn.init.xavier_normal_(h)).to(glob.device)
        init_hidden = Variable(torch.nn.init.normal_(h, mean=0, std=0.2).to(glob.device))

        c = torch.zeros(glob.num_layers*bi,batch_size,glob.hidden_dims)
        # init_cell = Variable(torch.nn.init.xavier_normal_(c)).to(glob.device)
        init_cell = Variable(torch.nn.init.normal_(c, mean=0, std=0.2).to(glob.device))

        # force model to learn initial states
        # init_hidden = nn.Parameter(torch.randn(glob.num_layers,batch_size,glob.hidden_dims).type(torch.FloatTensor), requires_grad=True).to(glob.device)
        # init_cell = nn.Parameter(torch.randn(glob.num_layers, batch_size, glob.hidden_dims).type(torch.FloatTensor),requires_grad=True).to(glob.device)

        return init_hidden,init_cell

    def forward(self, words_idx, batch_size, hidden, cell):
        """ batch_size words are fed in parallel to the network, with shape (1xbatch_size).
        After embedding lookup, the output has shape (1, batch_size, embedding_dim) """
        embeds = self.embedding_table(words_idx)
        embeds = embeds.view(1,batch_size, self.embedding_dim)

        output, (hidden,cell) = self.lstm(embeds, (hidden,cell))

        return output, hidden, cell



class SiameseLSTM(nn.Module):

    def __init__(self, vocab):
        super(SiameseLSTM, self).__init__()

        # Initialize one lstm, this will by default share variables across inputs
        self.lstm = LSTM(vocab)

        # Define loss function
        self.loss_function = nn.MSELoss()

        # Use Adam optimizer. Paper uses adadelta
        self.optimizer = optim.Adam(self.lstm.parameters(),
                                    lr=glob.learning_rate,
                                    betas=(glob.beta_1, 0.999))
        # self.optimizer = optim.SGD(params=self.lstm.parameters(),
        #                            lr = glob.learning_rate,
        #                            momentum=0.78,
        #                            weight_decay=0.0044,
        #                            nesterov=True)
        # self.optimizer = optim.Adadelta(params=self.lstm.parameters(),
        #                                 rho=0.95,
        #                                 eps=1e-6)

        # self.scheduler = ReduceLROnPlateau(optimizer=self.optimizer,
        #                                    mode='min',
        #                                    factor=0.1,
        #                                    threshold=0.000001,
        #                                    patience=2,
        #                                    verbose=True)
        self.scheduler = CyclicLR(optimizer=self.optimizer,
                                  base_lr=glob.learning_rate,
                                  max_lr=glob.learning_rate*10,
                                  step_size=5000)

    def forward(self, batch, is_singleton):
        """ Perform forward pass through the network. """

        # re-initialize hidden state and cell
        hidden_a_0, cell_a_0 = self.lstm.init_hidden(batch.s1.size(1))

        # run this batch through lstm_a
        # batches are of shape (sentence, num_batches)
        for t_i in range(batch.s1.size(0)):
            out_a, hidden_a_n, cell_a_n = self.lstm(words_idx=batch.s1[t_i, :],
                                                    batch_size = batch.s1.size(1),
                                                    hidden = hidden_a_0,
                                                    cell = cell_a_0)

        # run this batch through lstm_b
        hidden_b_0, cell_b_0 = self.lstm.init_hidden(batch.s2.size(1))
        for t_i in range(batch.s2.size(0)):
            out_b, hidden_b_n, cell_b_n = self.lstm(words_idx=batch.s2[t_i, :],
                                                    batch_size=batch.s2.size(1),
                                                    hidden=hidden_b_0,
                                                    cell=cell_b_0)

        # send final hidden states to 2d tensors
        hidden_a_n = hidden_a_n.squeeze()
        hidden_b_n = hidden_b_n.squeeze()

        # Get similarity predictions:
        # Clipping predictions with values taken from Mueller et al.'s implementation
        # test (|batch| = 1)prediction
        if is_singleton:
            return torch.exp(-torch.norm((hidden_a_n - hidden_b_n), 1))
        # train (|batch| > 1) prediction
        else:
            return torch.exp(-torch.norm((hidden_a_n - hidden_b_n), 1, 1))

    def get_loss(self, y_pred, y, is_test=False):
        ''' Compute MSE between predictions and scaled gold labels '''
        if is_test:
            return self.loss_function(y_pred,y)
        else:
            y = (y - 1) / 4.0
            # y_pred = torch.clamp(y_pred, min=1e-7,max=1.0 - 1e-7)
            return self.loss_function(y_pred, y)


