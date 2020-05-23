import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.insert(1, './')
from dotdict import *
from torch.optim import SGD
from torchsummary import summary
from config import *
import os
from logger import Logger
import time
from torch.utils.data import Dataset, DataLoader


class RLDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, episode):
        self.episode = episode

    def __len__(self):
        return len(self.episode)

    def __getitem__(self, idx):

        return (self.episode[idx])

def conv3x3(num_in, num_out, stride=1):
    return nn.Conv2d(in_channels=num_in, out_channels=num_out, stride=stride, kernel_size=3, padding=1, bias=False)

def conv1x1(num_in, num_out, stride):
    return nn.Conv2d(in_channels=num_in, out_channels=num_out, stride=stride, kernel_size=3, padding=1, bias=False)

class ResBlock(nn.Module):
    def __init__(self, num_in, num_out, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(num_in, num_out, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(num_out, num_out, stride=stride)
        self.bn2 = nn.BatchNorm2d(num_out)
        
        self.downsample = False
        if num_in != num_out or stride != 1:
            self.downsample = True
            self.downsample_conv = conv3x3(num_in, num_out, stride=stride)
            self.downsample_bn = nn.BatchNorm2d(num_out)
            
    def forward(self, x):
        residual = x
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        if self.downsample:
            residual = self.downsample_conv(residual)
        # print(residual.shape)
        out+=residual
        out = self.relu(out)
        # print(out.shape)
        return out


class AlphaGoNNet(nn.Module):
    def __init__(self, res_layers, board_size, action_size, num_features, num_layers, num_channels=256):
        super(AlphaGoNNet, self).__init__()
        # print("\nPARAMS")
        # print(res_layers, board_size, action_size, num_features, num_layers, num_channels, "\n")
        # convolutional block
        self.conv = conv3x3(num_in=num_features, num_out=num_channels, stride=1)
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        # residual tower
        res_list = [ResBlock(num_channels, num_channels) for _ in range(num_layers)]
        self.res_layers = nn.Sequential(*res_list)
        # policy head
        self.p_conv = conv1x1(num_in=num_channels, num_out=2, stride=1)
        self.p_bn = nn.BatchNorm2d(num_features=2)
        self.p_fc = nn.Linear(2 * board_size ** 2, action_size)
        self.p_log_softmax = nn.LogSoftmax(dim=1)
        # value head
        self.v_conv = conv1x1(num_in=num_channels, num_out=1, stride=1)
        self.v_bn = nn.BatchNorm2d(num_features=1)
        self.v_fc1 = nn.Linear(board_size**2, num_channels)
        self.v_fc2 = nn.Linear(num_channels,1)
        self.tanh = nn.Tanh()
    
    def forward(self, state):
        # first conv block
        # print("input:", state.shape)
        out = self.bn(self.conv(state))
        # print("conv1, bn1: ",out.shape)
        out = self.relu(out)
        # print(out.shape)
        # print("^^^^^^^^^^^^^^^")
        # residual tower
        # print(self.res_layers)
        # print(out)
        out = self.res_layers(out)
        # print(out.shape)
        # policy network
        p_out = self.p_bn(self.p_conv(out))
        # print(p_out.shape)

        p_out = self.relu(p_out)
        p_out = p_out.view(p_out.size(0), -1)
        # print(p_out.shape)
        # print(self.p_fc)
        p_out = self.p_fc(p_out)
        p_out = self.p_log_softmax(p_out)
        # value network
        v_out = self.v_bn( self.v_conv(out))
        v_out = self.relu(v_out)
        v_out = v_out.view(v_out.size(0), -1)
        v_out = self.v_fc1(v_out)
        v_out = self.relu(v_out)
        v_out = self.v_fc2(v_out)
        v_out = self.tanh(v_out)
        # print("p_out, v_out: ", p_out.shape, v_out.shape)
        return p_out,v_out

args = dotdict({
    'cuda': torch.cuda.is_available(),
    'board_size': BOARD_SIZE,
    'action_size':NUM_ACTIONS,
    'num_features':NUM_FEATURES,
    'num_layers':NUM_LAYERS,
    'num_channels': 256,
    'epochs': NUM_EPOCHS,
    'batch_size': BATCH_SIZE,
    'mini_batch': 2048,
    'lr': 0.001,
    'dropout': 0.3,
    'momentum':0.9,
    'l2': 0.0001,
    'num_checkpoint':1000,
    'temp1':1,
    'epsilon':0.25,
    'eta':0.03
#     'total_games':500000,
#     'num_simulations':1600,
#     'num_games':400,
#     'threshold_win':0.55,
#     'num_selfplay': 2500,
})

class AlphaLoss(nn.Module):
    """
    (z - v) ** 2 --> MSE Loss
    (-pi * logp) --> Cross Entropy Loss
    z : self_play_winner
    v : winner
    pi : self_play_probas
    p : probas
    """
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, log_ps, vs, target_ps, target_vs):
        value_loss = torch.mean(torch.pow(vs - target_vs, 2))
        policy_loss = -torch.mean(torch.sum(target_ps * log_ps, 1))
        self.v_loss = value_loss
        self.p_loss = policy_loss
        return value_loss + policy_loss


class PolicyValueNet():
    def __init__(self, args):
        self.args = args
        self.nn = AlphaGoNNet(res_layers=args.board_size, board_size=args.board_size, action_size=args.action_size, num_features=args.num_features, num_channels=args.num_channels, num_layers=args.num_layers)
        self.nn = self.nn.double()
        self.writer = Logger('./utils_6/logs')
        self.step1 = 0
        self.step2 = 0
        # print("- - - PolicyValueNet - - - ")
        if args.cuda:
            self.nn.cuda()
            self.device = torch.device("cuda:0") 
        else:
            self.device = torch.device("cpu") 
        self.optimizer = SGD(self.nn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2)
        self.alpha_loss = AlphaLoss()
        # print("------------------------------------------------------------------------------------")
        # print(self.nn)
        # print("------------------------------------------------------------------------------------")
        

    def divide_chunks(self,episode): 
        for i in range(0, len(episode[0]), self.args.batch_size):
            yield (torch.from_numpy(np.array(episode[0][i:i + self.args.batch_size])), \
                    torch.from_numpy(np.array(episode[1][i:i + self.args.batch_size])),\
                    torch.from_numpy(np.array(episode[2][i:i + self.args.batch_size])))



    def train(self, episode):
        print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")
        print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")
        print("* * * * * * * * * * * * * * * Training the network * * * * * * * * * * * * * * *")
        print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")
        print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")
        # print(len(episode[0]), len(episode[1]), len(episode[2]))
        # print(episode[0][0].shape, episode[1][0].shape, episode[2][0])
        print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")
        # print(batches)
        # from IPython import embed; embed()

        dataset = RLDataset(episode)
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                    shuffle=True, num_workers=4)
        batches = []
        for i_batch, (s, p, v) in enumerate(dataloader):
            batches.append((s.to(self.device), p.to(self.device), v.to(self.device)))
            # from IPython import embed; embed()

            # print(sample_batched.shape)

        for epoch in range(self.args.epochs):

                        
            # batches = self.divide_chunks(episode)
            total_loss=0
            total_vloss = 0
            total_ploss = 0
            print("EPOCH : ", epoch)
            self.nn.train()
            ###########
            # take a batch of data
            # extract state, action, value pair as tensors from mcts episodes
            # pass to nn
            ###########
            for states, p_mcts, v_mcts in batches:
                # print(states.shape, p_mcts.shape, v_mcts.shape)
                # if self.args.cuda:
                #     states = states.cuda()
                #     p_mcts = policies.cuda()
                #     v_mcts = rewards.cuda()
                # print("------------------------------------------------------------------------------------")
                # print(summary(self.nn,states.shape))
                # print("------------------------------------------------------------------------------------")
                self.optimizer.zero_grad()
                # print("get output of nn")
                start_t = time.time()
                log_ps, vs = self.nn(states)
                loss = self.alpha_loss(log_ps, vs, p_mcts, v_mcts)
                v_loss, p_loss = self.alpha_loss.v_loss, self.alpha_loss.p_loss
                total_loss += loss
                total_vloss += v_loss
                total_ploss += p_loss
                print("Loss : ", loss)
                loss.backward()
                self.optimizer.step()
                start_t = time.time()
                info = {'value_loss': v_loss, 
                        'policy_loss': p_loss}
                self.step1 += 1
                for tag, value in info.items():
                    self.writer.scalar_summary(tag, value, self.step1 + 1)
            info = {'total_loss': 1.0*total_loss/(len(episode)), 
                    'total vloss': 1.0*total_vloss/(len(episode)),
                    'total ploss': 1.0*total_ploss/(len(episode))}
            self.step2 += 1
            for tag, value in info.items():
                self.writer.scalar_summary(tag, value, self.step2 + 1)
    
    def predict(self, state):
        state = torch.from_numpy(state).double().to(self.device)
        state = state.unsqueeze(0)
        self.nn = self.nn.eval()
        with torch.no_grad():
            log_ps, vs = self.nn(state)
        # print(log_ps.shape)
        # print(vs.shape)
        return np.exp(log_ps.squeeze(0).cpu().detach().numpy()), vs.squeeze(0).cpu().detach().numpy()

    def save_checkpoint(self, tag, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, tag + '-' + filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        torch.save({ 'state_dict' : self.nn.state_dict(), }, filepath)
    
    def load_checkpoint(self, tag, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, tag + '-' + filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nn.load_state_dict(checkpoint['state_dict'])
