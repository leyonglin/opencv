
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
'''
计算机视觉：Computer Vision
作用：和图像相关的技术总称---图片处理
人工智能领域的计算机视觉：让计算机理解图像的内容
灰度级：每个采样点从最暗到最亮的分级，通常使用8位图像，即256级
色彩空间：
  1. 三原色：RGB颜色空间（天蓝色：(135,206,235)）
  2. H:色相(0-360) S:饱和度(0-1) V:明亮程度(0-1)  三角锥形状
  3. YUV
  4. CMYK
  5. Lab
颜色空间转换：
'''
"""
# # 基本操作
# # 读取图像：1-彩色图像  0-灰度图像
# # 一般在彩色图像上进行标记，边沿识别等
# # 灰度图像一般计算机处理
# im=cv2.imread('./piture/linux.png',1)    
# print(type(im))
# print(im.shape)
# cv2.imshow('im',im)   #显示原图图像，第一个参数为当前图像变量名
# # 转换为灰度图像
# im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)           
# cv2.imshow('im_gray',im_gray)
# # cv2.imwrite('./piture/linux-new.png',im)  #保存图像

# cv2.waitKey()                # 阻塞，等待用户按下按键退出
# cv2.destroyAllWindows()        #销毁所有创建的窗口
"""
"""
# #图像通道操作
# im=cv2.imread('./piture/opencvlogo.jpg',1)
# cv2.imshow('im',im)
# # 取出蓝色通道，并显示（单通道图像，计算机会显示灰度图像）
# b = im[:,:,0]   # 0为第一个通道--蓝色通道
# cv2.imshow("b",b)
# # 将原图抹掉蓝色通道(置0，即蓝色会变成黑色),则只剩绿红两通道
# im[:,:,0]=0
# cv2.imshow("im-b0",im)
# # 对原图再抹去绿色通道（绿色会变成黑色），则只剩红一个通道
# im[:,:,1]=0
# cv2.imshow('im-b0-g0',im)
# # 三通道全置0，则显示为黑色
# im[:,:,2]=0
# cv2.imshow('im-b0-g0-0',im)

# cv2.waitKey()                
# cv2.destroyAllWindows()       
"""
"""
# # 灰色图像直方图均衡化处理，会对亮度进行调整
# # 原图
# im=cv2.imread('./piture/sunrise.png',0)
# cv2.imshow('im',im)
# # 直方图均衡化处理
# im_equ = cv2.equalizeHist(im)
# cv2.imshow('im_equ',im_equ)

# # 原始图像直方图绘制
# plt.subplot(2,1,1)
# plt.hist(im.ravel(),      #返回一个扁平数组
#          256,[0,256],label='orig')
# plt.legend()
# # 处理后图像直方图绘制
# plt.subplot(2,1,2)
# plt.hist(im_equ.ravel(),      #返回一个扁平数组
#          256,[0,256],label='im_equ')
# plt.legend()
# plt.show()

# cv2.waitKey()               
# cv2.destroyAllWindows()  

# 彩色图像直方图均衡化处理，对其亮度进行调整
# 原图
im=cv2.imread('./piture/sunrise.png',1)
cv2.imshow('im',im)

# BGR==>YUV
yuv = cv2.cvtColor(im,cv2.COLOR_BGR2YUV)
# 取出亮度通道，进行均衡化处理，并将均衡化处理后的值赋值回原图像
yuv[...,0] = cv2.equalizeHist(yuv[...,0])    # ...表示取出行和列，0表示通道索引(亮度)

# YUV ==> BGR
equalized_color = cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)
cv2.imshow('equalized_color',equalized_color)
cv2.waitKey()               
cv2.destroyAllWindows()       
"""
"""
# 二值化与反二值化
im=cv2.imread('./piture/lena.png',0)
cv2.imshow('yuantu',im)

# 二值化，把大于阈值(127) 都变成255(白/亮)，把小于阈值(127) 都变成0(黑/暗)
t,rst = cv2.threshold(im,200,255,cv2.THRESH_BINARY)
cv2.imshow('erzhihua',rst)   #显示二值化图像
# # 与二值化相反
# t,rst2 = cv2.threshold(im,127,255,cv2.THRESH_BINARY_INV)
# cv2.imshow('反erzhi',rst2)   #显示反二值化图像

cv2.waitKey()                
cv2.destroyAllWindows()        
"""
"""
# # 仿射变换：图像形状，相对位置都不会发生变换
# # 图像翻转
# im=cv2.imread('./piture/linux.png',1)
# cv2.imshow('yuantu',im)

# # 0-垂直镜像
# im_flip0 = cv2.flip(im,0)
# cv2.imshow("chuizhijingxiang",im_flip0)

# # 1-水平镜像
# im_flip1 = cv2.flip(im,1)
# cv2.imshow("shuipingjingxiang",im_flip1)       

# # 平移
# def translate(im,x,y):
#     '''
#     对图像进行平移变换
#     :param im: 原始图像数据
#     :param x: 水平方向平移的像素
#     :param y: 垂直方向平移的像素
#     :return: 返回平移后的图像数据
#     '''
#     h,w = im.shape[:2]  #取出原图像高度，宽度
#     # 定义平移矩阵
#     M = np.float32([[1,0,x],
#                    [0,1,y]])
#     # 调用warpAffine函数实现平移变换
#     shifted = cv2.warpAffine(im,    #原始图像
#                              M,     #平移矩阵
#                              (w,h))  #输出图像大小
#     return shifted    

# # 旋转                      
# def rotate(im,angle,center=None,scale=1.0):
#     '''
#     对图像进行旋转变换
#     :param im: 原始图像数据
#     :param angle: 旋转角度
#     :param center: 旋转中心
#     :param scale：缩放比例
#     :return: 返回旋转后的图像数据
#     '''
#     h,w = im.shape[:2]  #取出原图像高度，宽度
#     # 计算旋转中心
#     if center is None:
#         center = (w/2,h/2)
#     # 生成旋转矩阵
#     M = cv2.getRotationMatrix2D(center,angle,scale)
#     # 调用warpAffine函数实现旋转变换
#     rotated = cv2.warpAffine(im,    #原始图像
#                              M,     #旋转矩阵
#                              (w,h))  #输出图像大小   
#     return rotated 

# if __name__=='__main__':
#     im=cv2.imread('./piture/linux.png',1)

#     # 向下移动50像素
#     shifted = translate(im,0,50)
#     cv2.imshow("xiangxiapingyi",shifted)
#     # 向左移动40像素，向下移动50像素
#     shifted = translate(im,-40,50)
#     cv2.imshow("zuoxiapingyi",shifted)

#     # 逆时针旋转45度
#     rotated = rotate(im,45)
#     cv2.imshow("nixuanzhuang45",rotated)
#     #顺时针旋转90度
#     rotated = rotate(im,-90)
#     cv2.imshow("shunxuanzhuang90",rotated)   

#     cv2.waitKey()               
#     cv2.destroyAllWindows()        
"""
"""
# 图像缩放
im=cv2.imread('./piture/linux.png',1)
cv2.imshow('im',im)

h,w = im.shape[:2]  #获取图像尺寸

dst_size = (int(w/2),int(h/2))  #宽度高度均为原来1/2
resized = cv2.resize(im,dst_size)  # 执行缩放
cv2.imshow("reduce",resized)

dst_size = (int(w*3/2),int(h*3/2))  #宽度高度均为原来3/2倍
method = cv2.INTER_NEAREST      #最邻近插值
resized = cv2.resize(im,dst_size,interpolation=method)  #执行放大
cv2.imshow("nearest",resized)

dst_size = (int(w*3/2),int(h*3/2))  #宽度高度均为原来3/2倍
method = cv2.INTER_LINEAR      #双线性插值，效果更好
resized = cv2.resize(im,dst_size,interpolation=method)  #执行放大
cv2.imshow("linear",resized)

cv2.waitKey()                
cv2.destroyAllWindows()       
"""
"""
# 图片裁剪，没有特殊的裁剪api，通过切片方式进行裁剪
# 图像随机裁剪
def random_crop(im,w,h):
    start_x = np.random.randint(0,im.shape[1])   # 裁剪起始x像素
    start_y = np.random.randint(0,im.shape[0])   # 裁剪起始y像素
    new_img = im[start_y:start_y+h,start_x:start_x+w]   #执行裁剪
    return new_img
# 图像中心裁剪
def center_crop(im,w,h):
    start_x = int(im.shape[1]/2)-int(w/2)  
    start_y = int(im.shape[0]/2)-int(h/2)
    new_img = im[start_y:start_y+h,start_x:start_x+w]
    return new_img

if __name__=='__main__':
    im=cv2.imread('./piture/banana.png',1)
    cv2.imshow('im',im)

    #随机裁剪
    new_img=random_crop(im,200,200)
    cv2.imshow("random_crop",new_img)

    #中心裁剪
    new_img2=center_crop(im,200,200)
    cv2.imshow("center_crop",new_img2)

    cv2.waitKey()               
    cv2.destroyAllWindows()       
"""
"""
# 图像相加，多副相同图像相加，实现图片降噪
a = cv2.imread('./piture/dust.webp',0)
b = cv2.imread('./piture/clock.webp',0)

dst1 = cv2.add(a,b)  # 图像直接相加，会导致图像越来越白/亮

#加权求和，水印效果
dst2 = cv2.addWeighted(a,0.6,   #图1及权重
                       b,0.4,   #图2及权重
                       0)       #亮度调节量
cv2.imshow('a',a)
cv2.imshow('b',b)
cv2.imshow('dst1',dst1)
cv2.imshow('dst2',dst2)
cv2.waitKey()                
cv2.destroyAllWindows()        
"""
"""
# 图像相减，消除背景，找出物体运动轨迹
a = cv2.imread('./piture/jihe1.png',0)
b = cv2.imread('./piture/jihe2.png',0)

dst = cv2.subtract(a,b)  # 图像相减

cv2.imshow('a',a)
cv2.imshow('b',b)
cv2.imshow('dst',dst)

cv2.waitKey()                # 阻塞，等待用户按下按键退出
cv2.destroyAllWindows()        #销毁所有创建的窗口
"""
"""
# 透视变换,常用于图像形状矫正
img = cv2.imread('./piture/yingshe.png')
rows,cols=img.shape[:2]
print(rows,cols)
cv2.imshow('img',img)
# 指定映射坐标
pst1 = np.float32([[58,2],[167,9],[8,196],[126,196]])  #输入图像四个顶点坐标
pst2 = np.float32([[16,2],[167,8],[8,196],[169,196]])  #输出图像四个顶点坐标

# 生成透视变换矩阵
M = cv2.getPerspectiveTransform(pst1,pst2)  
print(M.shape)
#执行透视变换，返回变换后的图像
dst = cv2.warpPerspective(img,
                           M,
                           (cols,rows)) #输出图像大小
cv2.imshow('dst',dst)

#生成透视变换矩阵
M = cv2.getPerspectiveTransform(pst2,pst1)
#执行变换
dst1 = cv2.warpPerspective(img,
                            M,
                            (cols,rows))   #输出图像大小
cv2.imshow('dst1',dst1)
cv2.waitKey()                
cv2.destroyAllWindows()        
"""
"""
#腐蚀，边缘收缩，将连在一起的图像分开
im = cv2.imread('./piture/fushi.png')
cv2.imshow('im',im)

kernel = np.ones((3,3),np.uint8)   #腐蚀核，即腐蚀范围
erosion = cv2.erode(im,        #原始图像
                    kernel,    #腐蚀核
                    iterations=3)  #迭代次数，值越大腐蚀越厉害
cv2.imshow('erosion',erosion)
cv2.waitKey()                
cv2.destroyAllWindows()      
"""
"""
#膨胀，边缘扩张，将没连在一起的图像连在一起
im = cv2.imread('./piture/pengzhang.png')
cv2.imshow('im',im)

kernel = np.ones((3,3),np.uint8)   #膨胀核
dilation = cv2.dilate(im,        #原始图像
                    kernel,    #膨胀核
                    iterations=5)  #迭代次数
cv2.imshow('dilation',dilation)
cv2.waitKey()                
cv2.destroyAllWindows()        
"""
"""
# 开运算，先腐蚀再膨胀，去噪
a = cv2.imread('./piture/kai1.png')
b = cv2.imread('./piture/kai2.png')
# 执行开运算
k = np.ones((10,10),np.uint8)
r1=cv2.morphologyEx(a,cv2.MORPH_OPEN,k)
r2=cv2.morphologyEx(b,cv2.MORPH_OPEN,k)

cv2.imshow('a',a)
cv2.imshow('b',b)
cv2.imshow('r1',r1)
cv2.imshow('r2',r2)

cv2.waitKey()                
cv2.destroyAllWindows()        
"""
"""
# 闭运算，先膨胀后腐蚀，降噪
a = cv2.imread('./piture/bi1.png')
b = cv2.imread('./piture/b2.png')
# 执行闭运算
k = np.ones((10,10),np.uint8)
r1=cv2.morphologyEx(a,cv2.MORPH_CLOSE,k,iterations=2)
r2=cv2.morphologyEx(b,cv2.MORPH_CLOSE,k,iterations=2)

cv2.imshow('a',a)
cv2.imshow('b',b)
cv2.imshow('r1',r1)
cv2.imshow('r2',r2)

cv2.waitKey()               
cv2.destroyAllWindows()        
"""
"""
#形态学梯度：膨胀减腐蚀的图像，提取图像边缘
a = cv2.imread('./piture/pengzhang.png')
k=np.ones((3,3),np.uint8)  #计算核
r=cv2.morphologyEx(a,cv2.MORPH_GRADIENT,k)
cv2.imshow("a",a)
cv2.imshow('r',r)
cv2.waitKey()                
cv2.destroyAllWindows()       
"""

# 礼帽运算：原始图像减去其开运算图像的操作，可获取噪点......
# 黑帽运算：用闭运算图像减去原始图像的操作，可获得噪点......
# 图像梯度：图像变化越大，梯度越大，计算的是图像的边缘信息
#    模板运算：模板(滤波器)是n*n矩阵，称为模板尺寸，模板运算分为模板卷积和模板排序
#            模板排序：选取像素特定值(最大/最小/中位数)作为输出值
#        作用：加强(锐化)/减弱(模糊)图像像素和像素之间的差异
#        降噪和模糊：都是减弱像素和像素之间的差异

"""
#滤波(模糊化)
a = cv2.imread('./piture/lena.png')
cv2.imshow("a",a)
#中值滤波(5*5)
im_median_blur = cv2.medianBlur(a,5)
cv2.imshow("im_median_blur",im_median_blur)
#均值滤波(3*3)
im_mean_blur = cv2.blur(a,(3,3))
cv2.imshow("im_mean_blur",im_mean_blur)
# 高斯滤波：满足高斯分布的矩阵作为高斯核(中间大，边缘小)
im_guassian_blur = cv2.GaussianBlur(a,
                                    (5,5),  # 5*5
                                    3)      # 高斯核标准差
cv2.imshow("im_guassian_blur",im_guassian_blur)
# 自定义高斯核执行滤波计算
gaussian_blur = np.array([[1,4,7,4,1],
                          [4,16,26,16,4],
                          [7,26,41,26,7],
                          [4,16,26,16,4],
                          [1,4,7,4,1]],np.float32) / 273
# 使用filter2D执行高斯滤波计算
im_guassian_blur2=cv2.filter2D(a,
                              -1,  #目标图像深度(通道数)，-1表示和原图像相同
                              gaussian_blur)  #滤波器
cv2.imshow("im_guassian_blur2",im_guassian_blur2)
        
#锐化
a = cv2.imread('./piture/lena.png',0)
cv2.imshow("a",a)
#锐化算子1
sharpen_1 = np.array([[-1,-1,-1],
                      [-1,9,-1],
                      [-1,-1,-1]])
im_sharpen1 = cv2.filter2D(a,-1,sharpen_1)
cv2.imshow("im_sharpen1",im_sharpen1)

#锐化算子1
sharpen_2 = np.array([[0,-1,0],
                      [-1,8,-1],
                      [0,-1,0]])/4.0
im_sharpen2 = cv2.filter2D(a,-1,sharpen_2)
cv2.imshow("im_sharpen2",im_sharpen2)
cv2.waitKey()               
cv2.destroyAllWindows()       
"""

# 边缘：不连续/非闭合的图形
# 轮廓：连续/闭合的整体图形
# 查找轮廓函数：cv2.findContours()
# img,cnts,hie=cv2.findContours(image,mode,method)
# image:原始灰度图像(黑色为背景)
# mode：
#     cv2.RETR_EXTERNA: 只检测外轮廓
#     cv2.RETR_LIST：对检测到的轮廓不建立等级关系
#     cv2.RETR_CCOMP：层次化
#     cv2.RETR_TREE：等级树结构的轮廓
# method：
#     cv2.CHAIN_APPROX_NONE：存储所有轮廓点
#     cv2.CHAIN_APPROX_SIMPLE：只保留线段起始点

"""
# 查找轮廓
im = cv2.imread('./piture/lunkuoyuantu.png')
cv2.imshow('im',im)
# 灰度化处理
im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#二值化
ret,im_binary=cv2.threshold(im_gray,127,255,cv2.THRESH_BINARY)
cv2.imshow('im_binary',im_binary)
# 查找轮廓，返回轮廓信息
img,cnts,hie=cv2.findContours(im_binary,    #经过二值化处理后的原图
                              cv2.RETR_EXTERNAL,  #只检测外轮廓
                              cv2.CHAIN_APPROX_NONE)  #存储所有轮廓点
# print(type(cnts))
# for cnt in cnts:
#     print(cnt.shape)

#绘制轮廓
im_cnt = cv2.drawContours(im,         #原图
                          cnts,       #轮廓数据
                          -1,         #绘制所有轮廓
                          (0,0,255),  #轮廓颜色，红色
                          2)          #轮廓粗细
cv2.imshow('im_cnt',im_cnt)
cv2.waitKey()                
cv2.destroyAllWindows()  
"""
"""
# 轮廓拟合
# 矩形包围圈
im = cv2.imread('./piture/lunkuo.png')
cv2.imshow('im',im)
# 灰度化处理
im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# 二值化处理
ret,im_binary = cv2.threshold(im_gray,127,255,cv2.THRESH_BINARY)
# 查找轮廓，返回轮廓信息
img,contours,hierarchy = cv2.findContours(im_binary,
                                        cv2.RETR_LIST,          #不建立等级关系
                                        cv2.CHAIN_APPROX_NONE)  #存储所有的轮廓点
print('contours[0].shape',contours[0].shape)

#返回轮廓定点及边长
x,y,w,h = cv2.boundingRect(contours[0])  #计算矩形包围框的xywh

# 绘制矩形包围框
brcnt = np.array([[[x,y]],[[x+w,y]],[[x+w,y+h]],[[x,y+h]]])
cv2.drawContours(im,             #绘制图像
                [brcnt],         #轮廓点列表
                -1,              #绘制全部轮廓
                (255,255,255),   #轮廓颜色：白色
                2)               #轮廓粗细
cv2.imshow('im1',im)
cv2.waitKey()                
cv2.destroyAllWindows()  
"""
"""
# 圆形包围圈
im = cv2.imread('./piture/lunkuo.png')
cv2.imshow('im',im)
# 灰度化处理
im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# 二值化处理
ret,im_binary = cv2.threshold(im_gray,127,255,cv2.THRESH_BINARY)
# 返回轮廓信息
img,contours,hierarchy = cv2.findContours(im_binary,
                                        cv2.RETR_LIST,          #不建立等级关系
                                        cv2.CHAIN_APPROX_NONE)  #存储所有的轮廓点
(x,y),radius = cv2.minEnclosingCircle(contours[0])   #产生轮廓的最小外接圆参数
print((x,y),radius)
center = (int(x),int(y))  #将圆心的坐标转换为整型
radius = int(radius)  #将半径转换为整型
cv2.circle(im,center,radius,(255,255,255),2)  #绘制圆形
cv2.imshow("im1",im)

cv2.waitKey()                
cv2.destroyAllWindows()  
"""
"""
# 绘制最优拟合椭圆
im = cv2.imread('./piture/lunkuo.png')
cv2.imshow('im',im)
# 灰度化处理
im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# 二值化处理
ret,im_binary = cv2.threshold(im_gray,127,255,cv2.THRESH_BINARY)
# 返回轮廓信息
img,contours,hierarchy = cv2.findContours(im_binary,
                                        cv2.RETR_LIST,          #不建立等级关系
                                        cv2.CHAIN_APPROX_NONE)  #存储所有的轮廓点
ellipse = cv2.fitEllipse(contours[0])   #拟合最优椭圆
print('ellipse',ellipse)
cv2.ellipse(im,ellipse,(0,0,255),2)  #绘制椭圆
cv2.imshow("im1",im) 

cv2.waitKey()                
cv2.destroyAllWindows()  
"""
"""
# 构建多边形，逼近轮廓
im = cv2.imread('./piture/lunkuo.png')
cv2.imshow('im',im)
im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# 二值化处理
ret,im_binary = cv2.threshold(im_gray,127,255,cv2.THRESH_BINARY)
# 返回轮廓信息
img,contours,hierarchy = cv2.findContours(im_binary,
                                        cv2.RETR_LIST,          #不建立等级关系
                                        cv2.CHAIN_APPROX_NONE)  #存储所有的轮廓点
# 精度1
adp = im.copy()    #图像复制
epsilon = 0.005 * cv2.arcLength(contours[0],True)   # 精度，根据周长计算
approx = cv2.approxPolyDP(contours[0],epsilon,True) # 构造多边形，返回多边形数据，true表示多边形是封闭的
adp = cv2.drawContours(adp,[approx],0,(0,0,255),2)  #绘制多边形 
cv2.imshow('adp',adp)
# 精度2
adp2 = im.copy()
epsilon = 0.01 * cv2.arcLength(contours[0],True)   # 精度，根据周长计算
approx = cv2.approxPolyDP(contours[0],epsilon,True) # 构造多边形
adp2 = cv2.drawContours(adp,[approx],0,(0,0,255),2)  #绘制多边形 
cv2.imshow('adp2',adp2)
cv2.waitKey()               
cv2.destroyAllWindows() 
"""
"""
# 图像校正
im = cv2.imread('./piture/toushibianhuan.png')
cv2.imshow('im',im)
# 灰度处理
im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

# # 二值化处理,效果不好
# ret,im_binary = cv2.threshold(im_gray,180,255,cv2.THRESH_BINARY)
# cv2.imshow('im_binary',im_binary)

# # 边沿提取1,Sobel算子，效果不好
# sobel = cv2.Sobel(im_gray,cv2.CV_64F,1,1,ksize=5)
# cv2.imshow("sobel",sobel)
# # 边沿提取2---Laplacian算子,效果不好
# lap = cv2.Laplacian(im_gray,cv2.CV_64F)
# cv2.imshow("Laplacian",lap)
# 模糊化
blurred = cv2.GaussianBlur(im_gray,(5,5),0)
# 膨胀
dilate = cv2.dilate(blurred,(3,3))
# 边沿提取3---canny算子  这个效果可以才对其进行预处理--模糊化和膨胀
canny = cv2.Canny(im,50,240)
cv2.imshow("canny",canny)

# 轮廓检测
img,cnts,hie = cv2.findContours(canny.copy(),             #原始图像
                                cv2.RETR_EXTERNAL,        #只检测外轮廓
                                cv2.CHAIN_APPROX_SIMPLE)  #只保留轮廓终点坐标
# 绘制轮廓，能检测出基本轮廓
im_cnt = cv2.drawContours(im,cnts,-1,(0,0,255),2)
cv2.imshow("im_cnt",im_cnt)

docCnt = None
# 计算轮廓面积，排序
if len(cnts) > 0:
    cnts = sorted(cnts,                 #可迭代对象
                  key=cv2.contourArea,  #排序依据，计算轮廓面积，根据面积排序
                  reverse=True)         #逆序排列
    for c in cnts:    #遍历排序后的每个轮廓
        peri = cv2.arcLength(c,True) #计算封闭轮廓周长
        approx = cv2.approxPolyDP(c,0.02*peri,True) #多边形拟合
        # 拟合出的第一个四边形认为是纸张的轮廓
        if len(approx) == 4:
            docCnt = approx
            break
# 绘制找到的四边形的交点
points = []
for peak in docCnt:
    peak = peak[0] #取出坐标
    # 绘制角点
    cv2.circle(im,              #绘制的图像
               tuple(peak),10,  #绘制圆形线条颜色，粗细
               (0,0,255),2)     #坐标添加到列表
    points.append(peak)  #坐标添加到列表
# cv2.imshow("im_point",im)
#根据勾股定理计算宽度，高度，再做透视变换
h = int(math.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2))  # 宽度
w = int(math.sqrt((points[2][0] - points[1][0])**2 + (points[2][1] - points[1][1])**2))  # 高度
print("w",w,"h",h)
src = np.float32([points[0],points[1],points[2],points[3]])
dst = np.float32([[0,0],[0,h],[w,h],[w,0]])
m = cv2.getPerspectiveTransform(src,dst)   #生成透视变换矩阵
result = cv2.warpPerspective(im_gray.copy(),m,(w,h))  #透视变换
cv2.imshow("im_result",result)

cv2.waitKey()               
cv2.destroyAllWindows() 
"""
"""
# 芯片瑕疵检测
im = cv2.imread('./piture/xinpian.png')
cv2.imshow('im',im)
# 灰度化
im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
cv2.imshow("im_gray",im_gray)
# 二值化
ret, im_bin = cv2.threshold(im_gray,162,255,cv2.THRESH_BINARY)
cv2.imshow("im_bin",im_bin)
# 提取轮廓，实心化填充
img,cnts,hie = cv2.findContours(im_bin,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
mask = np.zeros(im_bin.shape,np.uint8)  #创建值全为0的矩阵，形状和im_bin一致
im_fill = cv2.drawContours(mask,cnts,-1,(255,0,0),-1) #绘制轮廓并进行实心填充
cv2.imshow("im_fill",im_fill)
# 图像减法，找出瑕疵区域
im_sub = cv2.subtract(im_fill,im_bin)
cv2.imshow("im_sub",im_sub)
# 图像闭运算(先膨胀后腐蚀)，将离散的瑕疵点合并在一起
k = np.ones((10,10),np.uint8)
im_close = cv2.morphologyEx(im_sub,cv2.MORPH_CLOSE,k,iterations=3)
cv2.imshow("im_close",im_close)
# 提取瑕疵区域轮廓,绘制最小外接圆形
img,cnts,hie = cv2.findContours(im_close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
# print(cnts[1])
# 产生最小外接圆数据
(x,y),radius = cv2.minEnclosingCircle(cnts[0])
center = (int(x),int(y))
radius = int(radius)
cv2.circle(im_close,center,radius,(255,0,0),2)  #绘制瑕疵最小外接圆
cv2.imshow("im_circle",im_close)
# 在原始图像上绘制瑕疵
cv2.circle(im,center,radius,(0,0,255,2))
cv2.imshow('im_result',im)
# 计算外接圆形面积
area = math.pi * radius * radius
print("area:",area)
if area > 12:
    print('有瑕疵')
cv2.waitKey()                
cv2.destroyAllWindows()
"""

# #从摄像头获取帧数据
# #实例化VideoCapture对象，参数0表示第一个摄像头,如果是mp4等视频文件直接写绝对路径
# # cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('D:\\python\\opencv\\video.mp4')
# while cap.isOpened():  #摄像头处于打开状态
#     ret,frame = cap.read()  #捕获帧
#     cv2.imshow("frame",frame)
#     c = cv2.waitKey(1)  #等待1毫秒，等到用户敲击按键
#     if c == 27: #esc键
#         break
# cap.release()  #释放视频设备资源
# cv2.destroyAllWindows()


# #录制视频文件，两个过程：读取和写入
'''
编解码4字标记值说明
cv2.VideoWriter_fourcc('I','4','2','0')表示未压缩的YUV颜色编码格式，色度子采样为4:2:0
该编码格式具有较好的兼容性，但产生的文件较大，文件拓展名为.avi
cv2.VideoWriter_fourcc('P','I','M','I') 表示MPEG-1编码类型，生成的文件的扩展名为.avi
cv2.VideoWriter_fourcc('X','V','I','D') 表示MPEG-4编码类型，生成的文件的扩展名为.avi
cv2.VideoWriter_fourcc('T','H','E','O') 表示ogg vorbis编码类型，生成的文件的扩展名为.ogv
cv2.VideoWriter_fourcc('F','L','V','I') 表示Flash编码类型，生成的文件的扩展名为.flv
'''
# cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc('P','I','M','I')  #编解码4字标记值
# out = cv2.VideoWriter("output.avi",  #文件名
#                       fourcc,        #编解码类型
#                       20,            #fps(帧速度)
#                       (640,480))     #视频分辨率
# while cap.isOpened():
#     ret,frame = cap.read()  #读取帧
#     if ret == True:
#         out.write(frame)  #写入帧
#         cv2.imshow("frame",frame)
#         if cv2.waitKey(1) == 27:  #esc键
#             break
#     else:
#         break
# cap.release()  #释放视频设备资源
# out.release()
# cv2.destroyAllWindows()
