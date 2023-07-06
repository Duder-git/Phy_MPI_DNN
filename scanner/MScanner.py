import torch

from ConstantList import *
import math
import torch.nn.functional as F

from main_skip import device


# 扫描类
class MScannerClass:
    def __init__(self, VirtualPhantom,
                 SelectGradientX=2.0,
                 SelectGradientY=2.0,
                 DriveFrequencyX=2500000.0 / 102.0,
                 DriveFrequencyY=2500000.0 / 96.0,
                 Xn=128, Yn=128,
                 ExciteFrequency=2.5e4,
                 SampleFrequency=2.5e6,
                 PixSize=1e-4,
                 ):
        self._Phantom = VirtualPhantom

        self._CoilSensitivity = 1.0  # [T/A] 灵敏度

        self._Step = PixSize

        self._Gx = SelectGradientX / U0
        self._Gy = SelectGradientY / U0

        self._Xn = Xn
        self._Yn = Yn

        self._Ax = self._Xn * PixSize * self._Gx / 2 * 1.1
        self._Ay = self._Yn * PixSize * self._Gy / 2 * 1.1

        self._Xmax = self._Xn * PixSize / 2  # [m]
        self._Ymax = self._Yn * PixSize / 2  # [m]

        self._XSequence = torch.arange(-self._Xmax, self._Xmax, self._Step)
        self._YSequence = torch.arange(-self._Ymax, self._Ymax, self._Step)

        self._Fx = DriveFrequencyX
        self._Fy = DriveFrequencyY

        self._Fs = SampleFrequency
        self._Fex = ExciteFrequency

        self._TSequence = self._getTimes()
        self._Tn = self._TSequence.size(0)

        self._DHx, self._DirDHx = self._DriveStrength(self._Ax, self._Fx)
        self._DHy, self._DirDHy = self._DriveStrength(self._Ay, self._Fy)

        self._ffpX = self._DHx / self._Gx
        self._ffpY = self._DHy / self._Gy

        self._Vx = self._DirDHx / self._Gx
        self._Vy = self._DirDHy / self._Gy

        self._Phantom._Picture = self._Phantom._get_Picture(self._Xn, self._Yn)
        self._H = self.__get_FieldH()

    # 计算轨迹所需扫描时间
    def _getTimes(self):
        Nx = round(self._Fs / self._Fx)
        Ny = round(self._Fs / self._Fy)
        Nxy = (Nx * Ny / math.gcd(Nx, Ny))
        ScanTime = Nxy / self._Fs
        DT = 1 / self._Fs
        ScanT = torch.arange(0, ScanTime, DT)
        return ScanT

    # 计算磁场强度分布 [x,y,t]
    def __get_FieldH(self):
        GHx = self._Gx * self._XSequence.unsqueeze(0).unsqueeze(2)  # [A/m] [1,x,1]
        GHy = self._Gy * self._YSequence.unsqueeze(1).unsqueeze(2)  # [A/m] [y,1,1]

        DHx = self._DHx.unsqueeze(0).unsqueeze(0)
        DHy = self._DHy.unsqueeze(0).unsqueeze(0)

        Hx = -DHx + GHx
        Hy = -DHy + GHy

        HStrength = torch.sqrt(Hx ** 2 + Hy ** 2)
        return HStrength

    # 计算驱动场 及其导数
    def _DriveStrength(self, DriveAmplitude, DriveFrequency):
        DHx = DriveAmplitude * torch.cos(2.0 * PI * DriveFrequency * self._TSequence + PI / 2.0) * (-1.0)
        DeriDHx = DriveAmplitude * torch.sin(
            2.0 * PI * DriveFrequency * self._TSequence + PI / 2.0) * 2.0 * PI * DriveFrequency

        return DHx, DeriDHx

    # Initialize the phantom.
    def __init_Phantom(self):
        self._Phantom._Picture = self._Phantom._get_Picture(self._Xn, self._Yn)

    # 计算 电压信号
    def _GetVoltage_CPU(self, flag_Relaxation=False):
        self._Phantom._Picture = self._Phantom._get_Picture(self._Xn, self._Yn)
        self._H = self.__get_FieldH()

        phan = self._Phantom._Picture.unsqueeze(2)

        self._Ccoeffx = self._Phantom._cParticles * self._CoilSensitivity * self._Phantom._ms * self._Phantom._Bcoeff * self._DirDHx
        self._Ccoeffy = self._Phantom._cParticles * self._CoilSensitivity * self._Phantom._ms * self._Phantom._Bcoeff * self._DirDHy

        self._DLFTemp = self.__diffLangevin(self._Phantom._Bcoeff * self._H)

        RelaxationTime = self._Phantom._RelaxationTime

        SigX = self._DLFTemp * phan * self._Ccoeffx
        SigY = self._DLFTemp * phan * self._Ccoeffy

        VoltageX = torch.sum(SigX, dim=[0, 1])
        VoltageY = torch.sum(SigY, dim=[0, 1])

        if flag_Relaxation:
            VoltageX = self.__Relaxation(VoltageX, RelaxationTime)
            VoltageY = self.__Relaxation(VoltageY, RelaxationTime)

        Voltage = torch.sqrt(VoltageX ** 2 + VoltageY ** 2)
        return Voltage

    def __diffLangevin(self, inn):
        epsilon = 1e-20  # 设置一个较小的常数避免除以0
        inn = inn + epsilon
        out = 1 / (inn ** 2) - 1 / (torch.sinh(inn) ** 2)
        return out

    def __Relaxation(self, Signal, Rtime):
        TSequence = torch.arange(0, 50 / self._Fs, 1 / self._Fs)
        Kernel = (1e-6 / Rtime * torch.exp(-1 * TSequence / Rtime))
        Kernel = Kernel / sum(Kernel)
        RelaxationSignal = self.__ScanConv(Signal, Kernel)

        return RelaxationSignal

    def __ScanConv(self, Signal, ckernel):
        Signal = Signal.unsqueeze(0).unsqueeze(0)
        ckernel = ckernel.unsqueeze(0).unsqueeze(0)
        [x, y, z] = ckernel.size()
        cpadding = torch.zeros(1, 1, z - 1)
        Signal = torch.cat([cpadding, Signal], dim=2)
        ckernel = torch.flip(ckernel, dims=[2])
        Signal = F.conv1d(Signal, ckernel) / torch.sum(ckernel)
        return Signal.squeeze()

    # 计算 电压信号
    def _init_Voltage_GPU(self):
        self._Ccoeffx = self._Phantom._cParticles * self._CoilSensitivity * self._Phantom._ms * self._Phantom._Bcoeff * self._DirDHx.to(
            device)
        self._Ccoeffy = self._Phantom._cParticles * self._CoilSensitivity * self._Phantom._ms * self._Phantom._Bcoeff * self._DirDHy.to(
            device)

        self._DLFTemp = self.__diffLangevin(self._Phantom._Bcoeff * self._H).to(device)

    def _get_Voltage_GPU(self, phan, RelaxationTime=0):
        phan = phan.unsqueeze(2).to(device)

        SigX = self._DLFTemp * phan * self._Ccoeffx
        SigY = self._DLFTemp * phan * self._Ccoeffy

        VoltageX = torch.sum(SigX, dim=[0, 1]).to(device)
        VoltageY = torch.sum(SigY, dim=[0, 1]).to(device)

        if RelaxationTime != 0:
            VoltageX = self.__Relaxation_GPU(VoltageX, RelaxationTime)
            VoltageY = self.__Relaxation_GPU(VoltageY, RelaxationTime)

        Voltage = torch.sqrt(VoltageX ** 2 + VoltageY ** 2)

        return Voltage

    def __Relaxation_GPU(self, Signal, Rtime):
        TSequence = torch.arange(0, 50 / self._Fs, 1 / self._Fs).to(device)
        Kernel = (1e-6 / Rtime * torch.exp(-1 * TSequence / Rtime))
        Kernel = Kernel / sum(Kernel)
        RelaxationSignal = self.__ScanConv_GPU(Signal, Kernel)

        return RelaxationSignal

    def __ScanConv_GPU(self, Signal, ckernel):
        Signal = Signal.unsqueeze(0).unsqueeze(0)
        ckernel = ckernel.unsqueeze(0).unsqueeze(0)
        [x, y, z] = ckernel.size()
        cpadding = torch.zeros(1, 1, z - 1).to(device)
        Signal = torch.cat([cpadding, Signal], dim=2)
        ckernel = torch.flip(ckernel, dims=[2])
        Signal = F.conv1d(Signal, ckernel) / torch.sum(ckernel)
        return Signal.squeeze()
