import cv2
import numpy as np
import camera_configs
import open3d

cv2.namedWindow("depth")
cv2.createTrackbar("num", "depth", 3, 10, lambda x: None)
cv2.createTrackbar("blockSize", "depth", 21, 255, lambda x: None)

cap = cv2.VideoCapture("3492_14260.mp4")

iii = 0

while True:
    iii = iii+1
    ret1, frame = cap.read()

    # ret2, frame2 = camera2.read()
    # if not ret1 or not ret2:
    if not ret1:
        break

    frame1 = frame[0:399, 0:639]
    frame2 = frame[0:399, 639:1279]
    print(iii)
    if iii==59:
        cv2.imwrite('testimg.png',frame)

    # 根据更正map对图片进行重构
    img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)

    # 将图片置为灰度图，为StereoBM作准备
    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

    # 两个trackbar用来调节不同的参数查看效果
    num = cv2.getTrackbarPos("num", "depth")
    BlockSize = cv2.getTrackbarPos("blockSize", "depth")
    if BlockSize % 2 == 0:
        BlockSize += 1
    if BlockSize < 5:
        BlockSize = 5

    # 根据SGBM方法生成差异图
    stereo = cv2.StereoSGBM_create(numDisparities=16 * num, blockSize=BlockSize)

    disparity = stereo.compute(imgL, imgR)

    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 视差图梯度提取
    grad_x = cv2.Sobel(disp, cv2.CV_32F, 1, 0)  # 对x求一阶导
    grad_y = cv2.Sobel(disp, cv2.CV_32F, 0, 1)  # 对y求一阶导
    grad_xx = cv2.convertScaleAbs(grad_x)  # 用convertScaleAbs()函数将其转回原来的uint8形式
    grad_yy = cv2.convertScaleAbs(grad_y)
    grad_xy = cv2.addWeighted(grad_xx, 0.5, grad_yy, 0.5, 0)  # 图片融合
    _, im_at_fixed = cv2.threshold(grad_xy, 50, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("binary", im_at_fixed)

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
    # cv2.imshow("match", im_at_fixed)

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

    # 取四个方向的最小半径
    R = uu
    R = dd if dd < R else R
    R = ll if ll < R else R
    R = rr if rr < R else R

    cv2.circle(im_at_fixed, (y, x), R, (0, 0, 0), 2)
    cv2.imshow("land area", im_at_fixed)

    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
    threeD = np.zeros((400, 640), dtype=np.float32)
    threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16., camera_configs.Q)
    frame0 = cv2.hconcat([img1_rectified, img2_rectified])

    cv2.imshow("frame", frame0)

    #disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
    cv2.imshow("depth", disp)

    key = cv2.waitKey(1)
    if key == 27:
        break
    # 绘制三维点云
    if key == ord("s"):
        where_are_nan = np.isnan(threeD)
        where_are_inf = np.isinf(threeD)
        threeD[where_are_nan] = 0
        threeD[where_are_inf] = 0
        dst = np.zeros((400 * 640, 3), dtype=np.float32)
        color = np.zeros((400 * 640, 3), dtype=np.uint8)
        for i in range(0, 400):
            for j in range(0, 640):
                num = i * 640 + j
                dst[num] = threeD[i][j]
                color[num] = img1_rectified[i, j]
        # threeD.flatten()  #行展开
        # dst = np.zeros(threeD.shape, dtype=np.float32)
        # cv2.normalize(threeD, dst=dst, alpha=1.0, beta=0, norm_type=cv2.NORM_MINMAX)
        test_pcd = open3d.geometry.PointCloud()  # 定义点云
        test_pcd.points = open3d.utility.Vector3dVector(dst)  # 定义点云坐标位置
        test_pcd.colors = open3d.utility.Vector3dVector(color)  # 定义点云的颜色
        open3d.visualization.draw_geometries([test_pcd])

cap.release()
cv2.destroyAllWindows()
