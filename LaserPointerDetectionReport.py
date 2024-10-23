import cv2
import numpy as np

# 捕获摄像头视频
cap = cv2.VideoCapture(0)

while True:
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为HSV色彩空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 定义红色激光的HSV范围
    lower_red = np.array([160, 100, 100])
    upper_red = np.array([180, 255, 255])

    # 创建一个掩码，仅保留红色范围内的像素
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # 对掩码应用形态学操作以去除噪声
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 查找轮廓
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 计算轮廓的面积以忽略较小的假阳性
        area = cv2.contourArea(contour)
        if area > 100:
            # 绘制轮廓
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.putText(frame, "Laser Detected", (int(x-20), int(y-20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示结果帧
    cv2.imshow('Laser Detection', frame)

    # 按'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
