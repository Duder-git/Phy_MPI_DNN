from phantom import Phantom
import matplotlib.pyplot as plt


# 仿体验证
def test1():
    print("构建仿体类")
    PhanDir = 'D:\\document\\PyCharm\\Phy_MPI_DNN\\phantom\\phan_M.png'
    Phan = Phantom.PhantomClass(PhanDir)
    plt.figure()
    plt.imshow(Phan._Picture)
    plt.title("仿体图像")
    plt.show()


test1()
