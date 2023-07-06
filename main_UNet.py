import torch

from Unet import *
from models import *
from scanner import MScanner
from phantom import Phantom
import os
from scipy.io import savemat
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_data(file_path, title, data):
    with open(file_path, 'a') as file:
        file.write(f"{title}: {data}\n")


# 网络训练 返回浓度分布
def closure(scanner, Voltage, IterNums, SavePath, CrNet, RteNet, flag_Relaxation=False, LrCr=1e-3, LrRte=3e-5, Lambda=1):
    CrNet.train()

    # 网络参数
    ParametersCr = CrNet.parameters()  # 返回网络权重
    ParametersRte = RteNet.parameters()

    # 优化器
    OptimizerCr = torch.optim.Adam(ParametersCr, lr=LrCr)  # 优化器优化 自动调整模型参数以最小化损失函数
    OptimizerRte = torch.optim.Adam(ParametersRte, lr=LrRte)

    # 添加到GPU
    CrNet.to(device)
    RteNet.to(device)

    # 网络输入
    InputCr = torch.rand([1, 1, 128, 128]).to(device)
    InputRte = 10*torch.rand([1, 1, 1, 100]).to(device)

    # 输出
    Phan_iter = torch.rand([1, 1, 128, 128])

    # L1损失函数
    L1Loss = torch.nn.L1Loss()

    # 初始化迭代使用数据
    scanner._init_Voltage_GPU()

    # 检查文件是否存在
    file_path = SavePath + 'm_r.txt'
    if os.path.exists(file_path):
        os.remove(file_path)

    for Iter in range(IterNums):
        # 初始化梯度
        OptimizerCr.zero_grad()
        OptimizerRte.zero_grad()

        # 前向传播
        Phan_iter = CrNet(InputCr).squeeze()
        if flag_Relaxation:
            Rt = RteNet(InputRte).squeeze()
        else:
            Rt = torch.zeros(1).squeeze()

        # loss计算
        Loss = L1Loss(Voltage.to(device), scanner._get_Voltage_GPU(Phan_iter, RelaxationTime=Rt))
        #Loss += Lambda * Phan_iter.abs().sum().item()

        # 后向传播
        Loss.backward()

        # 参数优化 更新模型参数
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

            title = 'rt_' + str(Iter + 1)
            save_data(file_path, title, data=Rt.cpu().detach().numpy())

    return Phan_iter


# 深度学习验证
def main():
    print("构建仿体类")
    PhanDir = 'D:\\document\\PyCharm\\Phy_MPI_DNN\\phantom\\phan_MPI.png'
    Phan = Phantom.PhantomClass(PhanDir, RelaxationTime=2e6)
    flag_Relaxation = False

    print("构建扫描轨迹类")
    scanner = MScanner.MScannerClass(Phan)

    print("计算初始电压")
    Voltage = scanner._GetVoltage_CPU(flag_Relaxation=flag_Relaxation)

    print("模型设置 CrNet 浓度分布估计")
    CrNet = Unet()

    print("模型设置 RteNet 弛豫时间估计")
    RteNet = FCnet()

    # 迭代次数
    IterNums = 10000

    # 重建结果保存路径
    SavePath = os.getcwd() + '\\result\\'

    print("网络训练")
    c = closure(scanner, Voltage, IterNums, SavePath, CrNet, RteNet, flag_Relaxation)


if __name__ == '__main__':
    main()
