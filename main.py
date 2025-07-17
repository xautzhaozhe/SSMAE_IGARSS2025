import os
import numpy as np
import torch
import torch.optim as Optim
import scipy.io as sio
from SSMNet import SSMNet
from utils import get_params, seed_dict, ROC_AUC, Mahalanobis, Residual
import random
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Mydata import Data

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
data_dir = './data/'
save_dir = './results/'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def should_stop(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True
        return False


def main(file, i, j):
    # *********************************************************************************************************
    # set random seed
    seed = seed_dict[file]
    set_seed(seed)
    print(file)
    data_path = data_dir + file + '.mat'
    save_subdir = os.path.join(save_dir, file)
    if not os.path.exists(save_subdir):
        os.makedirs(save_subdir)
    # load data
    mat = sio.loadmat(data_path)
    img_var = mat['data']
    gt = mat['map']
    row, col, band = img_var.shape
    # *********************************************************************************************************

    LR = 1e-2  #
    end_iter = 15
    window_size = i
    spa_rate = 0.1
    spe_rate = 0.1
    batch_size = 2 ** j
    hidden_node = 50
    # *********************************************************************************************************
    data_set = Data(img=img_var, w_size=window_size, spatial_mask_rate=spa_rate, spectral_mask_rate=spe_rate)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, drop_last=False)
    # *********************************************************************************************************
    net = SSMNet(in_ch=band, out_ch=band, head_ch=hidden_node, w_size=window_size).to(device)
    mse = torch.nn.MSELoss(reduction='mean')
    p = get_params(net)
    optimizer = Optim.Adam(p, lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    early_stopper = EarlyStopper(patience=5, min_delta=0.001)
    print('Starting optimization')

    # **********************************************************************************************************
    # start train
    start = time.time()
    for iter in range(1, end_iter + 1):
        running_loss = 0.0
        net.train()
        loss = 0
        for idx, batch_data in enumerate(data_loader):
            code_center, window, mask_window = batch_data
            net_out = net(mask_window.to(device), code_center.to(device))
            loss = mse(net_out.cpu(), window)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            scheduler.step()
            torch.save(net, './checkpoint/{}_model.pth'.format(file))
        # if iter % 10 == 0:
        print('Epoch:', iter, '| train loss: %.4f' % loss.data.cpu().numpy())
        epoch_loss = running_loss / batch_size
        if early_stopper.should_stop(epoch_loss):
            print(f'Early stopping triggered at epoch {iter + 1}')
            break

    # *****************************************************************************************************
    # start test
    infer_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False, drop_last=False)
    infer_res_list = []
    for idx, data in enumerate(infer_loader):
        code_center, window, mask_window = data
        # inference
        with torch.no_grad():
            net = torch.load('./checkpoint/{}_model.pth'.format(file))
            infer_out = net(window.to(device), code_center.to(device))
            infer_res_list.append(infer_out.cpu())

    infer_out = torch.cat(infer_res_list, dim=0)
    out = infer_out[:, :, int(window_size / 2), int(window_size / 2)]
    out = out.detach().numpy()
    re_img = np.reshape(out, (row, col, band))
    # running time
    end = time.time()
    print("Runtime：%.2f" % (end - start))

    # ****************************************************************************************************
    img_var = (img_var - np.min(img_var)) / (np.max(img_var) - np.min(img_var))
    re_img = (re_img - np.min(re_img)) / (np.max(re_img) - np.min(re_img))
    result = Mahalanobis((img_var - re_img))
    sio.savemat('./results/{}/{}_rx-result.mat'.format(file, file), {'result': result})
    print('Detection result：')
    auc = ROC_AUC(result, gt)
    plt.imshow(result)
    plt.title('{}'.format(file))
    plt.savefig('./results/{}/{}-{}.png'.format(file, file, auc))

    return


if __name__ == "__main__":

    for file in ['Pavia_100', 'abu-beach-1']:
        i, j = 0, 0
        if file == 'Pavia_100':
            i, j = 19, 10
        elif file == 'abu-beach-1':
            i, j = 9, 12
        main(file, i, j)
