import torch.nn as nn


class SSMNet(nn.Module):

    def __init__(self, in_ch=189, out_ch=189, head_ch=32, w_size=7):
        super().__init__()
        n1 = int(in_ch / 2)

        # 设计浅层特征映射块，将输入 HSI 映射到特征空间
        self.Encoder1 = nn.Sequential(nn.Conv2d(in_ch, n1, kernel_size=1),
                                      nn.BatchNorm2d(n1),
                                      nn.LeakyReLU())

        self.Encoder2 = nn.Sequential(nn.Conv2d(n1, head_ch, kernel_size=1),
                                      nn.BatchNorm2d(head_ch),
                                      nn.Sigmoid())
        # body
        self.Conv1x1 = nn.Sequential(nn.Conv2d(head_ch, head_ch, kernel_size=1),
                                     nn.BatchNorm2d(head_ch),
                                     nn.LeakyReLU(inplace=True))

        self.Conv3x3 = nn.Sequential(nn.Conv2d(head_ch, head_ch, kernel_size=3, padding=1, padding_mode='reflect'),
                                     nn.BatchNorm2d(head_ch),
                                     nn.LeakyReLU())

        self.Conv5x5 = nn.Sequential(nn.Conv2d(head_ch, head_ch, kernel_size=5, padding=2, padding_mode='reflect'),
                                     nn.BatchNorm2d(head_ch),
                                     nn.LeakyReLU(inplace=True))

        self.fuse = nn.Sequential(nn.Conv2d(head_ch, head_ch, kernel_size=3, padding=1, padding_mode='reflect'),
                                  nn.BatchNorm2d(head_ch),
                                  nn.LeakyReLU())

        # 调制网络
        self.FC1 = nn.Sequential(nn.Conv2d(in_ch, n1, kernel_size=1),
                                 nn.BatchNorm2d(n1),
                                 nn.LeakyReLU())

        self.FC2 = nn.Sequential(nn.Conv2d(n1, head_ch, kernel_size=1),
                                 nn.BatchNorm2d(head_ch),
                                 nn.ReLU())

        # 解码网络，恢复背景
        self.Decoder1 = nn.Sequential(nn.Conv2d(head_ch, n1, kernel_size=1),
                                      nn.BatchNorm2d(n1),
                                      nn.LeakyReLU())

        self.Decoder2 = nn.Sequential(nn.Conv2d(n1, out_ch, kernel_size=1),
                                      nn.BatchNorm2d(out_ch), )

    def forward(self, x, center):
        x1 = self.Encoder1(x)
        x2 = self.Encoder2(x1)

        s1 = self.Conv1x1(x2)
        s2 = self.Conv3x3(x2)
        s3 = self.Conv5x5(x2)
        muti_scale = self.fuse(s1 + s2 + s3) + x2

        fc1_out = self.FC1(center)
        fc2_out = self.FC2(fc1_out)

        out1 = self.Decoder1(muti_scale + muti_scale * fc2_out)
        out2 = self.Decoder2(out1 + out1 * fc1_out)

        return out2
