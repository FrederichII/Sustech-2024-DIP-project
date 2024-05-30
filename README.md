# Sustech-2024-DIP-project

该项目使用的车牌数据集是CPPD2019，中科大的中国车牌开源数据集，下载方式可参考[[深度学习\] CCPD车牌数据集介绍_ccpd数据集-CSDN博客](https://blog.csdn.net/LuohenYJ/article/details/117752120)，体积较大，约12G，包含了各种条件下非常丰富的车牌图像。


预处理：输入一张包含车牌的图片，输出车牌中的每个字符的图像，以便后续做模板匹配。车牌定位的算法是基于彩色图像的阈值分割和边缘检测，鲁棒性比较差，输入的图像建议为正面拍摄的含车牌照片，车牌占画面比例不宜过小，否则可能会漏检测，车不要是蓝色或和车牌相近的颜色的，尽可能保证一定的清晰度，噪声过多也会影响输出质量。

调用方法：
```python
    import new_locate
    new_locate.preprocess()
```
需要去new_locate.py里面改一下输入以及输出的文件路径
如果遇到问题想查看preprocess函数产生的中间过程，调用时开启debugMode：
```python
   new_locate.preprocess(debugMode=True) 
```
函数会将中间产生的图像都show出来，按键盘任意键可切换下一张图片

做模板检测的时候，最好先用debug模式确认一下分割的字符串图像是否正确、是否合适用来作为模板匹配的输入



