import numpy as np
from scipy.interpolate import griddata
import torch
import matplotlib.pyplot as plt

from scanner import MScanner
from phantom import Phantom


def plt_3D_fig(x_data, y_data, z_data, x_label='X', y_label='Y', z_label='Z'):
    y_data = -y_data
    fig = plt.figure()

    ax = plt.subplot(projection='3d')
    ax.plot_trisurf(x_data, y_data, z_data, cmap='viridis', linewidth=0.9)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    plt.show()


def reConstruct_PixSize(ffpX, ffpY, Img, PixSize=2e-4):
    ffpX = ffpX.numpy()
    ffpY = ffpY.numpy()

    Xmax = max(ffpX)  # [m]
    Ymax = max(ffpY)  # [m]

    XSequence = np.arange(-Xmax, Xmax + PixSize, PixSize)
    YSequence = np.arange(-Ymax, Ymax + PixSize, PixSize)

    xpos, ypos = np.meshgrid(XSequence, YSequence, indexing='xy')
    ImgTan = griddata((ffpX, ffpY), Img, (xpos, ypos), method='linear')

    ImgTan[np.isnan(ImgTan)] = 0
    ImgTan = ImgTan / np.max(ImgTan)

    return ImgTan


def reConstruct_Pixel(ffpX, ffpY, Img, Pixel=61):
    ffpX = ffpX.numpy()
    ffpY = ffpY.numpy()

    Xmax = max(ffpX)  # [m]
    Ymax = max(ffpY)  # [m]

    PixSize = (2 * Xmax) / (Pixel - 1)
    XSequence = np.arange(-Xmax, Xmax + PixSize, PixSize)
    YSequence = np.arange(-Ymax, Ymax + PixSize, PixSize)

    xpos, ypos = np.meshgrid(XSequence, YSequence, indexing='xy')
    ImgTan = griddata((ffpX, ffpY), Img, (xpos, ypos), method='linear')

    ImgTan[np.isnan(ImgTan)] = 0
    ImgTan = ImgTan / np.max(ImgTan)

    return ImgTan


def test4():
    print("构建仿体类")
    PhanDir = 'D:\\document\\PyCharm\\Phy_MPI_DNN\\phantom\\phan_M.png'
    Phan = Phantom.PhantomClass(PhanDir)

    print("构建扫描轨迹类")
    scanner = MScanner.MScannerClass(Phan)

    U = scanner._GetVoltage_CPU(flag_Relaxation=False)
    v = torch.sqrt(scanner._Vx ** 2 + scanner._Vy ** 2)

    Sig = U / v
    plt.figure()
    plt.plot(U)
    plt.plot(v)
    plt.show()

    plt_3D_fig(scanner._ffpX, scanner._ffpY, Sig)

    Img = reConstruct_PixSize(scanner._ffpX, scanner._ffpY, Sig)
    plt.figure()
    plt.imshow(Img)
    plt.show()


test4()