import matplotlib.pyplot as plt
import torch.utils.data as data
import scipy.io as sio
import numpy as np
import torch
from einops import rearrange


# V2采用的是一次 mask, 然后训练
def standard(x):
    max_value = np.max(x)
    min_value = np.min(x)
    if max_value == min_value:
        return np.zeros_like(x)
    return (x - min_value) / (max_value - min_value)


def cosin_similarity(x, y):
    x_norm = np.sqrt(np.sum(x ** 2, axis=1))
    y_norm = np.sqrt(np.sum(y ** 2, axis=1))
    x_y_multi = np.sum(np.multiply(x, y), axis=1)
    return x_y_multi / (x_norm * y_norm + 1e-8)


def patch_encoded(patch, center):
    p_h, p_w, b = patch.shape
    patch_unfold = np.reshape(patch, [-1, b], order='F')
    assert patch_unfold.shape[1] == center.shape[1]
    encoded_weight = cosin_similarity(patch_unfold, center)

    encoded_weight = np.exp(encoded_weight) / np.sum(np.exp(encoded_weight))
    encoded_weight = encoded_weight[:, None]
    encoded_vector = np.sum(encoded_weight * patch_unfold, axis=0)
    encoded_vector = encoded_vector[None, :]
    return encoded_vector


class Data(data.Dataset):
    def __init__(self, img, w_size=85, spatial=True, spatial_mask_rate=0.2, spectral=True, spectral_mask_rate=0.2):
        self.w_size = w_size
        self.pad_size = w_size // 2
        self.spatial_mask_rate, self.spectral_mask_rate = spatial_mask_rate, spectral_mask_rate
        self.spatial, self.spectral = spatial, spectral
        self.h, self.w, self.b = img.shape
        self.nums = self.h * self.w
        img = standard(img)

        self.data = np.pad(img, ((self.pad_size, self.pad_size), (self.pad_size, self.pad_size), (0, 0)),
                           mode='reflect')

        # 空间掩码
        self.mask_data = self.data
        hh, ww, cc = self.data.shape
        self.spa_mask_out, self.mask_out = np.zeros(shape=(hh, ww)), np.zeros(shape=(hh, ww))
        if self.spatial:
            num_sample = hh * ww
            spa_mask = np.hstack([np.zeros(int(num_sample * self.spatial_mask_rate)),
                                  np.ones(num_sample - int(num_sample * self.spatial_mask_rate))])
            np.random.shuffle(spa_mask)
            spa_mask = np.reshape(spa_mask, (hh, ww))
            spa_mask = np.expand_dims(spa_mask, 2).repeat(self.b, axis=2)
            self.spa_mask_out = self.mask_data * spa_mask

        # 光谱掩码
        if self.spectral:
            spe_mask = np.hstack([np.zeros(int(cc * self.spectral_mask_rate)),
                                  np.ones(cc - int(cc * self.spectral_mask_rate))])
            np.random.shuffle(spe_mask)
            spe_mask = np.expand_dims(spe_mask, (0, 1))
            self.mask_out = self.spa_mask_out * spe_mask

    def __getitem__(self, index):

        position_y = index // self.w
        position_x = index - position_y * self.w
        position_x = position_x + self.pad_size
        position_y = position_y + self.pad_size

        windows_out = self.data[position_y - self.pad_size:position_y + self.pad_size + 1,
                      position_x - self.pad_size:position_x + self.pad_size + 1, :]
        center = windows_out[self.pad_size, self.pad_size]
        center = center[None, :]
        patch = windows_out
        coded_vector = patch_encoded(patch, center)  # 获取没有掩码之前的中心coded向量

        mask_windows_out = self.mask_out[position_y - self.pad_size:position_y + self.pad_size + 1,
                           position_x - self.pad_size:position_x + self.pad_size + 1, :]

        return (torch.unsqueeze(torch.from_numpy(coded_vector).float().permute(1, 0), dim=2),
                torch.from_numpy(windows_out).float().permute(2, 0, 1),
                torch.from_numpy(mask_windows_out).float().permute(2, 0, 1))

    def __len__(self):
        return self.nums


if __name__ == '__main__':

    img = sio.loadmat('./data/HYDICE.mat')['data']
    print(img.shape)
    data = Data(img)

    infer_res_list = []
    for i in range(8000):
        center, coded_vector, window, mask_window = data.__getitem__(5676)
        plt.imshow(mask_window.T[:, :, 5])
        # plt.plot(mask_window.T[:, 4, 5])
        plt.show()
        # infer_res_list.append(mask_window[:, 2, 2])

    # infer_out = torch.cat(infer_res_list, dim=0)
    # out = infer_out.detach().numpy()
    # data = np.reshape(out, (80, 100, 175))
    # print(data.shape)
    #
    # sio.savemat('gt_re-data.mat', {'data': data})
