import torch
from models import *
from scanner import MScanner
from phantom import Phantom
import os
from scipy.io import savemat
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 网络训练 返回浓度分布
def closure(scanner, Voltage, IterNums, SavePath, CrNet, RteNet, LrCr=1e-3, LrRte=3e-5, Lambda=1.0):
    CrNet.train()

    # 网络参数
    ParametersCr = CrNet.parameters()
    ParametersRte = RteNet.parameters()

    # 优化器
    OptimizerCr = torch.optim.Adam(ParametersCr, lr=LrCr)
    OptimizerRte = torch.optim.Adam(ParametersRte, lr=LrRte)

    CrNet.to(device)
    RteNet.to(device)

    # 网络输入
    InputCr = torch.rand([1, 1, 128, 128]).to(device)
    InputRte = 10 * torch.rand([1, 1, 1, 100]).to(device)
    # 输出
    Phan_iter = torch.rand([1, 1, 128, 128])
    # L1损失函数
    L1Loss = torch.nn.L1Loss()
    # 初始化迭代使用数据
    scanner._init_Voltage_GPU()

    for Iter in range(IterNums):

        # 梯度置0
        OptimizerCr.zero_grad()
        OptimizerRte.zero_grad()

        # 前向传播
        Phan_iter = CrNet(InputCr).squeeze()
        Rt = RteNet(InputRte).squeeze()

        # loss计算
        Loss = L1Loss(Voltage.to(device), scanner._get_Voltage_GPU(Phan_iter, RelaxationTime=Rt))
        Loss += Lambda * sum(abs(Phan_iter))

        # 后向传播
        Loss.backward()

        # 参数优化
        OptimizerCr.step()
        OptimizerRte.step()

        # 输出loss保存重建数据
        print((Iter + 1), 'loss:', Loss)
        if (Iter + 1) % 100 == 0:
            ImgName = SavePath + 'Img\\' + 'Img' + str(Iter + 1) + '.jpeg'
            img = Phan_iter.cpu().detach().numpy()
            cv2.imwrite(ImgName, img * 255)

            CrName = SavePath + 'Data\\' + 'm_c_' + str(Iter + 1) + '.mat'
            savemat(CrName, {'c': Phan_iter.cpu().detach().numpy()})

            RtName = SavePath + 'Rt\\' + 'm_r_' + str(Iter + 1) + '.mat'
            savemat(RtName, {'rt': Rt.cpu().detach().numpy()})

    return Phan_iter


# 深度学习验证
def main():
    print("构建仿体类")
    PhanDir = 'D:\\document\\PyCharm\\Phy_MPI_DNN\\phantom\\phan_M.png'
    Phan = Phantom.PhantomClass(PhanDir)
    print("构建扫描轨迹类")
    scanner = MScanner.MScannerClass(Phan)
    print("计算初始电压")
    Voltage = scanner._GetVoltage_CPU()

    print("模型设置 CrNet 浓度分布估计")
    CrNet = skip(1, 1,
                 num_channels_down=[8, 16, 32, 64, 128],
                 num_channels_up=[8, 16, 32, 64, 128],
                 num_channels_skip=[0, 0, 0, 4, 4],
                 upsample_mode='bilinear',
                 need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')

    print("模型设置 RteNet 弛豫时间估计")
    RteNet = FCnet()

    # 迭代次数
    IterNums = 5000

    # 重建结果保存路径
    SavePath = os.getcwd() + '\\result\\'

    # 学习率
    LrCr = 1e-3
    LrRte = 3e-5
    # l1正则化系数
    Lambda = 1.0

    print("网络训练")
    c = closure(scanner, Voltage, IterNums, SavePath, CrNet, RteNet)


if __name__ == '__main__':
    main()
