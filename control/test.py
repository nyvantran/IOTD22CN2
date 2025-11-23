import cv2
import numpy as np

from lane_detection import ConfigHelper


def main():
    img_path = "img.png"
    img = cv2.imread(img_path)
    warp_helper = ConfigHelper(img, 4, window_name="Warp Point Configuration")
    warp_point = np.array(warp_helper.get_points())
    # x, y = cv2.norm(warp_point[0], warp_point[1]), cv2.norm(warp_point[0], warp_point[1])
    # x, y = int(x), int(y)
    x, y = 600, 600
    # x1, y1 = warp_point[0][0], warp_point[0][1]
    x1, y1 = 0, 0
    w, h = img.shape[1], img.shape[0]
    warp_dst = np.array([[x1, y1], [x, y1], [x, y], [x1, y]])
    homography = cv2.findHomography(warp_point, warp_dst)
    print("Homography Matrix:\n", homography)
    for point in warp_point:
        img = cv2.circle(img, (point[0], point[1]), 1, (0, 0, 255), 1)

    new_img = cv2.warpPerspective(img, homography[0], (x, y))
    cv2.imshow("Original", img)
    cv2.imshow("Warped", new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
