# readme

## 结构
```
Phy_MPI_DNN
    |-- models
        |-- skip.py         Unet网络 用于计算图像数据
        |-- FCnet.py        计算弛豫时间
    |-- phantom
        |-- phan_M.png      M形状仿体
        |-- phan_MPI.png    MPI字母仿体
        |-- Phantom.py      仿体类，处理仿体相关数据
    |-- result 
        |-- Img             存放迭代图像
        |-- Data            存放.mat结果
        |-- Rt.txt          弛豫时间
    |-- scanner
        |-- MScanner.py     扫描轨迹类,处理扫描轨迹计算电压信号
    |-- verification
        |-- show_mat.py     读取mat展示结果
        |-- test_Phan.py    测试仿体类
        |-- test_Scanner.py 测试扫描类
        |-- test_torch.py   测试网络
        |-- test_ReImg.py   测试重建图像是否正常 
```

## 说明

目标：基于物理模型约束的MPI重建图像

过程：

    网络模型
        -- 输入随机图像
        -- 输出重建图像
    物理模型约束
        输入实测电压信号
        基于迭代图像计算电压信号
        计算电压信号与输入信号L1范数作为损失函数
        反向传递损失函数，更新网络
        计算新的迭代图像

结果：实现从信号的重建图像的两步走，从随机图像到信号——从信号到图像




