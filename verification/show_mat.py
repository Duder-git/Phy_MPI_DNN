
import matplotlib.pyplot as plt
import scipy.io

ReadPath = 'D:\\document\\PyCharm\\Phy_MPI_DNN\\result\\'
filename = 'm_C_1100.mat'
ImgName = ReadPath + filename

mat_data = scipy.io.loadmat(ImgName)
image_data = mat_data['c']

# 显示图像
plt.figure
plt.imshow(image_data)
plt.axis('off')
plt.show()

