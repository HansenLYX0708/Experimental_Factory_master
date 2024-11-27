import math
import os
import cv2
import numpy as np
from skimage import filters


def brenner(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-2):
    	for y in range(0, shape[1]):
            out+=(int(img[x+2,y])-int(img[x,y]))**2
    return out

def Laplacian(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    return cv2.Laplacian(img,cv2.CV_64F).var()

def SMD(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
    	for y in range(1, shape[1]):
            out+=math.fabs(int(img[x,y])-int(img[x,y-1]))
            out+=math.fabs(int(img[x,y]-int(img[x+1,y])))
    return out

def SMD2(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out+=math.fabs(int(img[x,y])-int(img[x+1,y]))*math.fabs(int(img[x,y]-int(img[x,y+1])))
    return out

def variance(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    out = 0
    u = np.mean(img)
    shape = np.shape(img)
    for x in range(0,shape[0]):
        for y in range(0,shape[1]):
            out+=(img[x,y]-u)**2
    return out

def energy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out+=((int(img[x+1,y])-int(img[x,y]))**2)+((int(img[x,y+1]-int(img[x,y])))**2)
    return out

def Vollath(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    shape = np.shape(img)
    u = np.mean(img)
    out = -shape[0]*shape[1]*(u**2)
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]):
            out+=int(img[x,y])*int(img[x+1,y])
    return out

def entropy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    out = 0
    count = np.shape(img)[0]*np.shape(img)[1]
    p = np.bincount(np.array(img).flatten())
    for i in range(0, len(p)):
        if p[i]!=0:
            out-=p[i]*math.log(p[i]/count)/count
    return out

def Tenengrad(img):
    tmp = filters.sobel(img)
    source=np.sum(tmp**2)
    out=np.sqrt(source)
    # strp3: 绘制图片并保存  不应该写在这里  抽象出来   这是共有的部分
    return out



# -*-coding=UTF-8-*-
"""
在无参考图下，检测图片质量的方法
"""


class BlurDetection:
    def __init__(self, strDir):
        print("图片检测对象已经创建...")
        self.strDir = strDir

    def _getAllImg(self, strType='jpg'):
        """
        根据目录读取所有的图片
        :param strType: 图片的类型
        :return:  图片列表
        """
        names = []
        for root, dirs, files in os.walk(self.strDir):  # 此处有bug  如果调试的数据还放在这里，将会递归的遍历所有文件
            for file in files:
                # if os.path.splitext(file)[1]=='jpg':
                names.append(str(file))
        return names

    def _imageToMatrix(self, image):
        """
        根据名称读取图片对象转化矩阵
        :param strName:
        :return: 返回矩阵
        """
        imgMat = np.matrix(image)
        return imgMat

    def _blurDetection(self, imgName):

        # step 1 图像的预处理
        img2gray, reImg = self.preImgOps(imgName)
        imgMat=self._imageToMatrix(img2gray)/255.0
        x, y = imgMat.shape
        score = 0
        for i in range(x - 2):
            for j in range(y - 2):
                score += (imgMat[i + 2, j] - imgMat[i, j]) ** 2
        # step3: 绘制图片并保存  不应该写在这里  抽象出来   这是共有的部分
        score=score/10
        newImg = self._drawImgFonts(reImg, str(score))
        newDir = self.strDir + "/_blurDetection_/"
        if not os.path.exists(newDir):
            os.makedirs(newDir)
        newPath = newDir + imgName
        cv2.imwrite(newPath, newImg)  # 保存图片
        cv2.imshow(imgName, newImg)
        cv2.waitKey(0)
        return score

    def _SMDDetection(self, imgName):

        # step 1 图像的预处理
        img2gray, reImg = self.preImgOps(imgName)
        f=self._imageToMatrix(img2gray)/255.0
        x, y = f.shape
        score = 0
        for i in range(x - 1):
            for j in range(y - 1):
                score += np.abs(f[i+1,j]-f[i,j])+np.abs(f[i,j]-f[i+1,j])
        # strp3: 绘制图片并保存  不应该写在这里  抽象出来   这是共有的部分
        score=score/100
        newImg = self._drawImgFonts(reImg, str(score))
        newDir = self.strDir + "/_SMDDetection_/"
        if not os.path.exists(newDir):
            os.makedirs(newDir)
        newPath = newDir + imgName
        cv2.imwrite(newPath, newImg)  # 保存图片
        cv2.imshow(imgName, newImg)
        cv2.waitKey(0)
        return score

    def _SMD2Detection(self, imgName):
        """
        灰度方差乘积
        :param imgName:
        :return:
        """
        # step 1 图像的预处理
        img2gray, reImg = self.preImgOps(imgName)
        f=self._imageToMatrix(img2gray)/255.0
        x, y = f.shape
        score = 0
        for i in range(x - 1):
            for j in range(y - 1):
                score += np.abs(f[i+1,j]-f[i,j])*np.abs(f[i,j]-f[i,j+1])
        # strp3: 绘制图片并保存  不应该写在这里  抽象出来   这是共有的部分
        score=score
        newImg = self._drawImgFonts(reImg, str(score))
        newDir = self.strDir + "/_SMD2Detection_/"
        if not os.path.exists(newDir):
            os.makedirs(newDir)
        newPath = newDir + imgName
        cv2.imwrite(newPath, newImg)  # 保存图片
        cv2.imshow(imgName, newImg)
        cv2.waitKey(0)
        return score
    def _Variance(self, imgName):
        """
               灰度方差乘积
               :param imgName:
               :return:
               """
        # step 1 图像的预处理
        img2gray, reImg = self.preImgOps(imgName)
        f = self._imageToMatrix(img2gray)

        # strp3: 绘制图片并保存  不应该写在这里  抽象出来   这是共有的部分
        score = np.var(f)
        newImg = self._drawImgFonts(reImg, str(score))
        newDir = self.strDir + "/_Variance_/"
        if not os.path.exists(newDir):
            os.makedirs(newDir)
        newPath = newDir + imgName
        cv2.imwrite(newPath, newImg)  # 保存图片
        cv2.imshow(imgName, newImg)
        cv2.waitKey(0)
        return score
    def _Vollath(self,imgName):
        """
                       灰度方差乘积
                       :param imgName:
                       :return:
                       """
        # step 1 图像的预处理
        img2gray, reImg = self.preImgOps(imgName)
        f = self._imageToMatrix(img2gray)
        source=0
        x,y=f.shape
        for i in range(x-1):
            for j in range(y):
                source+=f[i,j]*f[i+1,j]
        source=source-x*y*np.mean(f)
        # strp3: 绘制图片并保存  不应该写在这里  抽象出来   这是共有的部分

        newImg = self._drawImgFonts(reImg, str(source))
        newDir = self.strDir + "/_Vollath_/"
        if not os.path.exists(newDir):
            os.makedirs(newDir)
        newPath = newDir + imgName
        cv2.imwrite(newPath, newImg)  # 保存图片
        cv2.imshow(imgName, newImg)
        cv2.waitKey(0)
        return source
    def _Tenengrad(self,imgName):
        """
                       灰度方差乘积
                       :param imgName:
                       :return:
                       """
        # step 1 图像的预处理
        img2gray, reImg = self.preImgOps(imgName)
        f = self._imageToMatrix(img2gray)

        tmp = filters.sobel(f)
        source=np.sum(tmp**2)
        source=np.sqrt(source)
        # strp3: 绘制图片并保存  不应该写在这里  抽象出来   这是共有的部分

        newImg = self._drawImgFonts(reImg, str(source))
        newDir = self.strDir + "/_Tenengrad_/"
        if not os.path.exists(newDir):
            os.makedirs(newDir)
        newPath = newDir + imgName
        cv2.imwrite(newPath, newImg)  # 保存图片
        cv2.imshow(imgName, newImg)
        cv2.waitKey(0)
        return source

    def Test_Tenengrad(self):
        imgList = self._getAllImg(self.strDir)
        for i in range(len(imgList)):
            score = self._Tenengrad(imgList[i])
            print(str(imgList[i]) + " is " + str(score))

    def Test_Vollath(self):
        imgList = self._getAllImg(self.strDir)
        for i in range(len(imgList)):
            score = self._Variance(imgList[i])
            print(str(imgList[i]) + " is " + str(score))


    def TestVariance(self):
        imgList = self._getAllImg(self.strDir)
        for i in range(len(imgList)):
            score = self._Variance(imgList[i])
            print(str(imgList[i]) + " is " + str(score))

    def TestSMD2(self):
        imgList = self._getAllImg(self.strDir)

        for i in range(len(imgList)):
            score = self._SMD2Detection(imgList[i])
            print(str(imgList[i]) + " is " + str(score))
        return
    def TestSMD(self):
        imgList = self._getAllImg(self.strDir)

        for i in range(len(imgList)):
            score = self._SMDDetection(imgList[i])
            print(str(imgList[i]) + " is " + str(score))
        return

    def TestBrener(self):
        imgList = self._getAllImg(self.strDir)

        for i in range(len(imgList)):
            score = self._blurDetection(imgList[i])
            print(str(imgList[i]) + " is " + str(score))
        return

    def preImgOps(self, imgName):
        """
        图像的预处理操作
        :param imgName: 图像的而明朝
        :return: 灰度化和resize之后的图片对象
        """
        strPath = self.strDir + imgName

        img = cv2.imread(strPath)  # 读取图片
        cv2.moveWindow("", 1000, 100)
        # cv2.imshow("原始图", img)
        # 预处理操作
        reImg = cv2.resize(img, (800, 900), interpolation=cv2.INTER_CUBIC)  #
        img2gray = cv2.cvtColor(reImg, cv2.COLOR_BGR2GRAY)  # 将图片压缩为单通道的灰度图
        return img2gray, reImg

    def _drawImgFonts(self, img, strContent):
        """
        绘制图像
        :param img: cv下的图片对象
        :param strContent: 书写的图片内容
        :return:
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontSize = 5
        # 照片 添加的文字    /左上角坐标   字体   字体大小   颜色        字体粗细
        cv2.putText(img, strContent, (0, 200), font, fontSize, (0, 255, 0), 6)

        return img

    def _lapulaseDetection(self, imgName):
        """
        :param strdir: 文件所在的目录
        :param name: 文件名称
        :return: 检测模糊后的分数
        """
        # step1: 预处理
        img2gray, reImg = self.preImgOps(imgName)
        # step2: laplacian算子 获取评分
        resLap = cv2.Laplacian(img2gray, cv2.CV_64F)
        score = resLap.var()
        print("Laplacian %s score of given image is %s", str(score))
        # strp3: 绘制图片并保存  不应该写在这里  抽象出来   这是共有的部分
        newImg = self._drawImgFonts(reImg, str(score))
        newDir = self.strDir + "/_lapulaseDetection_/"
        if not os.path.exists(newDir):
            os.makedirs(newDir)
        newPath = newDir + imgName
        # 显示
        cv2.imwrite(newPath, newImg)  # 保存图片
        cv2.imshow(imgName, newImg)
        cv2.waitKey(0)

        # step3: 返回分数
        return score

    def TestDect(self):
        names = self._getAllImg()
        for i in range(len(names)):
            score = self._lapulaseDetection(names[i])
            print(str(names[i]) + " is " + str(score))
        return
