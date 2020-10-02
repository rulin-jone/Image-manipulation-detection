# Image-manipulation-detection
基于论文Learning Rich Features for Image Manipulation Detection的学习与代码详解
代码原链接详见 https://github.com/LarryJiang134/Image_manipulation_detection  by LarryJiang134

另外，在过程中参考了https://blog.csdn.net/weixin_43380510/article/details/88544830 的解决手段

仓库目的为帮助大家实现和理解该代码。在配置过程中，不断修改了一些bug，这些bug是在源代码中readme没有提到的，因此提供给大家参考。
在修改过程中还使用tkinter添加了一个简单的界面系统，能够帮助更好的理解代码。

本代码实现环境：win10 + Tensor flow 1.8 + python 3.6，建议按照此配置进行环境安装。
代码中的bbox等文件是在win环境下编译的，如果要在Linux系统下运行项目，bbox等文件需要重新编译。

训练所需要的DIY_dataset数据集需要使用work工具自行生成，本例使用coco数据集生成，也可以换成其他数据集。
