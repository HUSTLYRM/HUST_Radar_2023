import cv2


def find_chessboard_corners(image, pattern_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        image_with_corners = cv2.drawChessboardCorners(image, pattern_size, corners, ret)
        return corners, image_with_corners
    else:
        return None, image


if __name__ == '__main__':
    # 设置棋盘格尺寸
    pattern_size = (8, 6)
    # 图像路径
    image_path = 'chessboard.jpg'

    # 进行棋盘格角点检测
    corners, image_with_corners = find_chessboard_corners(image_path, pattern_size)

    # 显示带有角点的图像
    cv2.imshow('Chessboard Corners', image_with_corners)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
