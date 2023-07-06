import torch
import matplotlib.pyplot as plt

from scanner import MScanner
from phantom import Phantom


def plt_2D_fig(x_data, y_data, x_label='X', y_label='Y'):
    plt.figure()

    plt.plot(x_data, y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()


# 扫描轨迹验证
def test2():
    print("构建仿体类")
    PhanDir = 'D:\\document\\PyCharm\\Phy_MPI_DNN\\phantom\\phan_M.png'
    Phan = Phantom.PhantomClass(PhanDir)

    print("构建扫描轨迹类")
    scanner = MScanner.MScannerClass(Phan)
    #scanner = MScanner.MScannerClass(Phan, DriveFrequencyX=25000, DriveFrequencyY=2500)

    plt_2D_fig(scanner._ffpX, scanner._ffpY)


# 执行程序
test2()
