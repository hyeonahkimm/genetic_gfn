import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from proxy.model.mlp import MLP


class DropoutRegressor(nn.Module):
    def __init__(self, tokenizer,num_token,max_len,args,device='cuda:0'):
        super().__init__()

        self.num_tokens = num_token
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.sigmoid = nn.Sigmoid()
        self.args = args
        self.device = device
        self.init_model()
        #self.eos_tok = 4

    def init_model(self):

        self.model = MLP(num_tokens=self.num_tokens,
                            num_outputs=1,
                            num_hid=self.args['proxy_nemb'],
                            num_layers=self.args['proxy_layers'], # TODO: add these as hyperparameters?
                            dropout=self.args['proxy_dropout'],
                            max_len=self.max_len)
        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), self.args['proxy_learning_rate'],
                            weight_decay=self.args['proxy_weight_decay'])
            
    def fit(self, data, reset=False):
        if reset:
            self.init_model()

        pbar = tqdm(range(self.args['proxy_num_iterations']))
        pbar.set_description('Proxy training')
        for i in pbar:
            x, y = data.sample(self.args['proxy_mbsize'])

            inp_x = F.one_hot(x.long(), num_classes=self.num_tokens+1)[:, :, :-1].to(torch.float32)
            inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
            inp[:, :inp_x.shape[1], :] = inp_x
            x = inp.reshape(x.shape[0], -1).to(self.device).detach()
            y = torch.tensor(y, device=self.device, dtype=torch.float).reshape(-1)

            output = self.model(x, None).squeeze(1)
            loss = (output - y).pow(2).mean()
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

            pbar.set_postfix(loss = loss.item())

        # losses = []
        # test_losses = []
        # best_params = None
        # best_loss = 1e6
        # early_stop_tol = 5
        # early_stop_count = 0
        # epoch_length = 100
        # if reset:
        #     self.init_model()
        

        # for it in tqdm(range(self.proxy_num_iterations), disable=False):
        #     x, y = data.sample(256)

        #     x = self.tokenizer.process(x).to(self.device)

        #     inp_x = F.one_hot(x, num_classes=self.num_tokens+1)[:, :, :-1].to(torch.float32)
        #     inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
        #     inp[:, :inp_x.shape[1], :] = inp_x
        #     x = inp.reshape(x.shape[0], -1).to(self.device).detach()
        #     y = torch.tensor(y, device=self.device, dtype=torch.float).reshape(-1)

        #     output = self.model(x, None).squeeze(1)
        #     loss = (output - y).pow(2).mean()
        #     loss.backward()
        #     self.opt.step()
        #     self.opt.zero_grad()
            
        #     losses.append(loss.item())

        #     if not it % epoch_length:
        #         vx, vy = data.validation_set()
        #         vlosses = []
        #         for j in range(len(vx) // 256):
        #             x = self.tokenizer.process(vx[j*256:(j+1)*256]).to(self.device)

        #             inp_x = F.one_hot(x, num_classes=self.num_tokens+1)[:, :, :-1].to(torch.float32)
        #             inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
        #             inp[:, :inp_x.shape[1], :] = inp_x
        #             x = inp.reshape(x.shape[0], -1).to(self.device).detach()
        #             y = torch.tensor(vy[j*256:(j+1)*256], device=self.device, dtype=torch.float).reshape(-1)

        #             output = self.model(x, None).squeeze(1)
        #             loss = (output -  y).pow(2)
        #             vlosses.append(loss.sum().item())

        #         test_loss = np.sum(vlosses) / len(vx)
        #         test_losses.append(test_loss)
        #         if test_loss < best_loss:
        #             best_loss = test_loss
        #             best_params = [i.data.cpu().numpy() for i in self.model.parameters()]
        #             early_stop_count = 0
        #         else:
        #             early_stop_count += 1

        #         if early_stop_count >= early_stop_tol:
        #             print(best_loss)
        #             print('early stopping')
        #             break

        # if 0:
        #     # Put best parameters back in
        #     for i, besti in zip(self.model.parameters(), best_params):
        #         i.data = torch.tensor(besti).to(self.device)

        # return {}

    def forward(self, x, uncertainty_call=False):
        # x = self.tokenizer.process(curr_x).to(self.device)
        x = x.long()

        inp_x = F.one_hot(x, num_classes=self.num_tokens+1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
        inp[:, :inp_x.shape[1], :] = inp_x
        x = inp.reshape(x.shape[0], -1).to(self.device).detach()
        if uncertainty_call:
            ys = self.model(x, None).unsqueeze(0)
        else:
            self.model.eval()

            ys = self.model(x, None)
            self.model.train()
        return ys
    
    def forward_with_uncertainty(self, x):
        self.model.train()
        with torch.no_grad():
            outputs = torch.cat([self.forward(x, True) for _ in range(25)])
        return outputs.mean(dim=0), outputs.std(dim=0)


    def eval(self, x):
        self.model.eval()
        with torch.no_grad():
            outputs = torch.cat([self.forward(x, True) for _ in range(25)])
        return outputs.mean(dim=0), outputs.std(dim=0)


    def save(self, path):
        torch.save({'model_state_dict': self.state_dict()}, path)
    
    def load(self, path):
        self.load_state_dict(path)


class EnsembleRegressor(nn.Module):
    def __init__(self, tokenizer,num_token,max_len,args,device='cuda:0'):
        super().__init__()

        self.num_tokens = num_token
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.sigmoid = nn.Sigmoid()
        self.args = args
        self.device = device
        self.init_model()
        #self.eos_tok = 4

    def init_model(self):

        self.models = [MLP(num_tokens=self.num_tokens,
                            num_outputs=1,
                            num_hid=self.args['proxy_nemb'],
                            num_layers=self.args['proxy_layers'], # TODO: add these as hyperparameters?
                            dropout=self.args['proxy_dropout'],
                            max_len=self.max_len) for i in range(self.args.proxy_num_dropout_samples)]
        [model.to(self.args.device) for model in self.models]
        self.params = sum([list(model.parameters()) for model in self.models], [])

        self.opt = torch.optim.Adam(self.params, self.args['proxy_learning_rate'],
                            weight_decay=self.args['proxy_weight_decay'])
            
    def fit(self, data, reset=False):
        if reset:
            self.init_model()

        pbar = tqdm(range(self.args['proxy_num_iterations']))
        pbar.set_description('Proxy training')
        for i in pbar:
            x, y = data.sample(self.args['proxy_mbsize'])

            inp_x = F.one_hot(x.long(), num_classes=self.num_tokens+1)[:, :, :-1].to(torch.float32)
            inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
            inp[:, :inp_x.shape[1], :] = inp_x
            x = inp.reshape(x.shape[0], -1).to(self.device).detach()
            y = torch.tensor(y, device=self.device, dtype=torch.float).reshape(-1)

            output = self.model(x, None).squeeze(1)
            loss = (output - y).pow(2).mean()
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

            pbar.set_postfix(loss = loss.item())


class EnsembleRegressor(nn.Module):
    def __init__(self, tokenizer,num_token,max_len):
        super().__init__()

        self.num_tokens = num_token
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.init_model()
        self.device = 'cuda:0'
        self.sigmoid = nn.Sigmoid()
        self.proxy_num_iterations = 50000
        

    def init_model(self):
        self.device = 'cuda:0'
        self.models = [MLP(num_tokens=self.num_tokens,
                                num_outputs=1,
                                num_hid=2048,
                                num_layers=2,
                                dropout=0.1,
                                max_len=self.max_len) for i in range(5)]
        [model.to(self.device) for model in self.models]
        self.params = sum([list(model.parameters()) for model in self.models], [])
        self.opt = torch.optim.Adam(self.params, 1e-4,
                            weight_decay=0.0001)

    def fit(self, data, reset=False):
        losses = []
        test_losses = []
        best_params = None
        best_loss = 1e6
        early_stop_tol = 100
        early_stop_count = 0
        epoch_length = 100
        if reset:
            self.init_model()

        for it in range(self.proxy_num_iterations):
            x, y = data.sample(256)
            x = self.tokenizer.process(x).to(self.device)
            y = torch.tensor(y, device=self.device, dtype=torch.float).reshape(-1)
            output = self._call_models(x).mean(0)
            loss = (output - y).pow(2).mean()
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            
            losses.append(loss.item())

            

            if not it % epoch_length:
                vx, vy = data.validation_set()
                vlosses = []
                for j in range(len(vx) // 256):
                    x = self.tokenizer.process(vx[j*256:(j+1)*256]).to(self.device)
                    y = torch.tensor(vy[j*256:(j+1)*256], device=self.device, dtype=torch.float).reshape(-1)

                    output = self._call_models(x).mean(0)
                    
                    loss = (output -  y).pow(2)
                    vlosses.append(loss.sum().item())

                test_loss = np.sum(vlosses) / len(vx)
                test_losses.append(test_loss)

                if test_loss < best_loss:
                    best_loss = test_loss
                    best_params = [[i.data.cpu().numpy() for i in model.parameters()] for model in self.models]
                    early_stop_count = 0
                else:
                    early_stop_count += 1

                if early_stop_count >= early_stop_tol:
                    print(best_loss)
                    print('early stopping')
                    break


            # Put best parameters back in
        for i, model in enumerate(self.models):
            for i, besti in zip(model.parameters(), best_params[i]):
                i.data = torch.tensor(besti).to(self.device)
        
        return {}
    
    def _call_models(self, x):
        x = self.tokenizer.process(x).to(self.device)

        inp_x = F.one_hot(x, num_classes=self.num_tokens+1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
        inp[:, :inp_x.shape[1], :] = inp_x
        x = inp.reshape(x.shape[0], -1).to(self.device).detach()

        ys = torch.cat([model(x, None).unsqueeze(0) for model in self.models])
        return ys
    
    def forward_with_uncertainty(self, x):
        with torch.no_grad():
            outputs = self._call_models(x)
        return outputs.mean(dim=0), outputs.std(dim=0)

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(path)