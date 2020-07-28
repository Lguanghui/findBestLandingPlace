from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication,QMainWindow,QMessageBox,QFileDialog
from GUI.myGUIWindow import Ui_mainWindow
import sys
import numpy as np
import camera_configs
import cv2

class mainWindow(QMainWindow,Ui_mainWindow):
    def __init__(self):
        #窗口初始化
        super(mainWindow,self).__init__()
        self.setupUi(self)
        self.timer = QTimer(self)   #定时器
        self.cap = cv2.VideoCapture()    #视频流
        self.meanArea = 0.0     #平均面积
        self.frameCount = 0     #视频帧数
        self.IniUI()

    def IniUI(self):
        #初始化
        self.rightEyeImg.clear()
        self.leftEyeImg.clear()
        self.depthImg.clear()
        self.resultImg.clear()
        self.pBtn_readEyeImg.clicked.connect(self.show_initialIMG)
        self.pBtn_getDepthImg.clicked.connect(self.show_depthIMG)
        self.pBtn_getBestArea.clicked.connect(self.show_resultIMG)
        self.pBtn_refresh.clicked.connect(self.clearAll)
        self.pBtn_dynamic.clicked.connect(self.show_dynamic)
        self.timer.timeout.connect(self.dynamic_showing)
        self.slider_BlockSize.setValue(21)
        self.slider_num.setValue(3)
        self.BlockSize = self.slider_BlockSize.value()
        self.Num = self.slider_num.value()
        self.label_blocksize.setText(str(self.BlockSize))
        self.label_num.setText(str(self.Num))
        self.slider_BlockSize.valueChanged.connect(self.slider_blocksize_changed)
        self.slider_num.valueChanged.connect(self.slider_num_changed)
        self.slider_num.sliderPressed.connect(self.slider_num_changed)

    def slider_blocksize_changed(self):
        '''监听blocksize改变'''
        self.BlockSize = self.slider_BlockSize.value()
        self.label_blocksize.setText(str(self.slider_BlockSize.value()))

    def slider_num_changed(self):
        '''监听num改变'''
        self.Num = self.slider_num.value()
        self.label_num.setText(str(self.slider_num.value()))

    def show_dynamic(self):
        '''开启/关闭动态显示'''
        if self.timer.isActive() == False:      #当按下按钮时，定时器未启动的话，启动定时器，显示图像
            path , ok = QFileDialog.getOpenFileName(self,'选取视频文件','.','(*.mp4 *.avi)')
            if ok:
                self.meanArea = 0.0  # 平均面积
                self.frameCount = 0  # 视频帧数
                ret = self.cap.open(path)
                if ret == False:
                    QMessageBox.warning(self, 'warning', "打开视频失败", buttons=QMessageBox.Ok)
                else:
                    self.timer.start(100)        #开始计时，间隔为30ms
                    self.pBtn_dynamic.setText('关闭动态显示')
            else:
                QMessageBox.warning(self, 'warning', "您需要选择一个视频文件哦😯", buttons=QMessageBox.Ok)
                return
        else:       #按下按钮时，已处在播放状态
            self.timer.stop()       #停止计时
            self.cap.release()
            self.clearAll()
            self.pBtn_dynamic.setText('开启动态显示')
            self.meanArea = 0.0
            self.frameCount = 0

    def dynamic_showing(self):
        '''动态显示'''
        ret,img = self.cap.read()
        if ret == False:
            self.timer.stop()  # 停止计时
            self.cap.release()
            self.clearAll()
            self.pBtn_dynamic.setText('开启动态显示')
            self.meanArea = 0.0
            self.frameCount = 0
            return
        self.show_initialIMG(img,1)
        self.show_depthIMG()
        self.show_resultIMG()

    def clearAll(self):
        '''清除显示的图片'''
        self.rightEyeImg.clear()
        self.leftEyeImg.clear()
        self.depthImg.clear()
        self.resultImg.clear()
        self.coordinate_x.clear()
        self.coordinate_y.clear()
        self.coordinate_z.clear()
        self.Area.clear()
        self.meanArea = 0.0  # 平均面积
        self.frameCount = 0  # 视频帧数

    def show_initialIMG(self,frame,flag = 0):
        '''显示左右目图片
        flag：0代表默认静态的图片，1代表动态显示
        frame：动态显示的图片'''
        # 读取图片
        self.meanArea = 0.0  # 平均面积
        self.frameCount = 0  # 视频帧数
        if flag == 1:
            img = frame
        else:
            path , ret = QFileDialog.getOpenFileName(self,'选取双目图像','.','(*.jpg *.png *.bmp *.jpeg)')
            if ret:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
            else:
                QMessageBox.warning(self, 'warning', "您需要选择一张图片哦😯", buttons=QMessageBox.Ok)
                return
        self.leftImg = img[0:399, 0:639]
        self.rightImg = img[0:399, 639:1279]

        # 根据更正map对图片进行重构
        self.leftImg = cv2.remap(self.leftImg, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
        self.rightImg = cv2.remap(self.rightImg, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)

        #反转颜色为RGB
        cv2.cvtColor(self.leftImg,cv2.COLOR_BGR2RGB,self.leftImg)
        cv2.cvtColor(self.rightImg,cv2.COLOR_BGR2RGB,self.rightImg)

        #尺寸
        h = self.leftEyeImg.height()
        w = self.rightEyeImg.width()
        channel = self.leftImg.shape[2]

        #重置尺寸适应label
        self.leftImg = cv2.resize(self.leftImg,(w,h))
        self.rightImg = cv2.resize(self.rightImg,(w,h))

        #使用QImage绘制图像
        QIMG_left = QImage(self.leftImg.data,w,h,channel*w,QImage.Format_RGB888)
        QIMG_right = QImage(self.rightImg.data,w,h,channel*w,QImage.Format_RGB888)

        pixmapL = QPixmap.fromImage(QIMG_left)
        pixmapR = QPixmap.fromImage(QIMG_right)

        self.leftEyeImg.setPixmap(pixmapL)
        self.rightEyeImg.setPixmap(pixmapR)

    def show_depthIMG(self):
        '''显示深度图'''
        self.get_depthIMG()
        # 尺寸
        h = self.leftEyeImg.height()
        w = self.rightEyeImg.width()
        channel = 3
        self.disp = self.disp[:,16*self.Num:639]

        #伪彩色处理
        disp = cv2.applyColorMap(self.disp, cv2.COLORMAP_JET)
        disp = cv2.resize(disp,(w,h))

        #显示
        QIMG_depthIMG = QImage(disp.data,w,h,w*channel,QImage.Format_RGB888)
        pixmap_depthIMG = QPixmap.fromImage(QIMG_depthIMG)
        self.depthImg.setPixmap(pixmap_depthIMG)

    def show_resultIMG(self):
        '''显示最终的结果图'''
        self.get_landArea()
        # 尺寸
        h = self.leftEyeImg.height()
        w = self.rightEyeImg.width()
        channel = self.landAreaImg.shape[2]

        self.landAreaImg = cv2.resize(self.landAreaImg,(w,h))

        QIMG_landAreaIMG = QImage(self.landAreaImg.data,w,h,w*channel,QImage.Format_RGB888)
        pixmap_landAreaIMG = QPixmap.fromImage(QIMG_landAreaIMG)
        self.resultImg.setPixmap(pixmap_landAreaIMG)
        #显示坐标和面积
        self.coordinate_y.setText(str(round(self.y,3)))
        self.coordinate_x.setText(str(round(self.x,3)))
        self.coordinate_z.setText(str(round(self.z,3)))
        self.Area.setText(str(round(3.14*(self.R**2),3)))

    def get_depthIMG(self):
        '''计算深度图'''
        #转化为灰度图
        self.leftImg_gray = cv2.cvtColor(self.leftImg,cv2.COLOR_RGB2GRAY)
        self.rightImg_gray = cv2.cvtColor(self.rightImg,cv2.COLOR_RGB2GRAY)

        # 根据SGBM方法生成差异图
        stereo = cv2.StereoSGBM_create(numDisparities=16 * self.Num, blockSize=self.BlockSize)
        self.disparity = stereo.compute(self.leftImg_gray,self.rightImg_gray)
        self.fornorm = self.disparity
        self.fornorm[self.fornorm < 0] = 0 # 上下限截断，显示图像亮度稳定
        self.fornorm[self.fornorm > 511] = 511
        self.disp = cv2.normalize(self.fornorm, self.fornorm, alpha=-500, beta=300, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    def get_landArea(self):
        '''计算最佳降落位置'''
        self.frameCount = self.frameCount + 1
        # 视差图梯度提取
        grad_x = cv2.Sobel(self.disparity, cv2.CV_32F, 1, 0)  # 对x求一阶导
        grad_y = cv2.Sobel(self.disparity, cv2.CV_32F, 0, 1)  # 对y求一阶导
        grad_xx = cv2.convertScaleAbs(grad_x)  # 用convertScaleAbs()函数将其转回原来的uint8形式
        grad_yy = cv2.convertScaleAbs(grad_y)
        grad_xy = cv2.addWeighted(grad_xx, 0.5, grad_yy, 0.5, 0)  # 图片融合
        _, im_at_fixed = cv2.threshold(grad_xy, 4, 255, cv2.THRESH_BINARY_INV)

        # 使用模板匹配查找平坦区域
        template = np.zeros((120, 120), dtype=np.uint8)
        for i in range(template.shape[0]):
            for j in range(template.shape[1]):
                template[i, j] = 255

        result = cv2.matchTemplate(im_at_fixed, template, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        tl = min_loc
        br = (tl[0] + 120, tl[1] + 120)
        # cv2.rectangle(im_at_fixed, tl, br, (0, 0, 0), 2)

        # 遍历查找到的中心点，绘制降落的圆形区域
        uu = 0
        x = tl[1] + 60
        y = tl[0] + 60
        for i in range(0, x - 1):
            if im_at_fixed[x - i, y] == 255:
                uu = i
            else:
                break

        dd = 0
        for i in range(1, im_at_fixed.shape[0] - x - 1):
            if im_at_fixed[x + i, y] == 255:
                dd = i
            else:
                break

        ll = 0
        for i in range(1, y - 1):
            if im_at_fixed[x, y - 1] == 255:
                ll = i
            else:
                break

        rr = 0
        for i in range(0, im_at_fixed.shape[1] - y - 1):
            if im_at_fixed[x, y + rr] == 255:
                rr = i
            else:
                break

        # 将图片扩展至3d空间中，其z方向的值则为当前的距离
        threeD = np.zeros((400, 640), dtype=np.float32)
        threeD = cv2.reprojectImageTo3D(self.disparity.astype(np.float32) / 16., camera_configs.Q)

        # 取四个方向的最小半径
        R = uu
        R = dd if dd < R else R
        R = ll if ll < R else R
        R = rr if rr < R else R
        im_at_fixed = cv2.cvtColor(im_at_fixed,cv2.COLOR_GRAY2RGB)

        # 设置显示坐标
        self.x = threeD[x, y, 0]
        self.y = threeD[x, y, 1]
        self.z = threeD[x, y, 2]
        self.R = R

        #阈值及均值优化
        meanArea = (3.14*(self.R**2) + self.meanArea*(self.frameCount -1 ))/self.frameCount     #当前均值
        if 3.14*(self.R**2) >= meanArea and 3.14*(self.R**2) > 7000:
            cv2.circle(im_at_fixed, (y, x), R, (255, 0, 0), 2)

        self.landAreaImg = im_at_fixed
        self.meanArea = meanArea        #均值更新


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = mainWindow()
    win.show()
    sys.exit(app.exec())

