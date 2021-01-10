import cv2

def filters(img):
    gray_scale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("grayscale", gray_scale_img)

    equal_hist_img = cv2.equalizeHist(gray_scale_img)
    cv2.imshow("equalizeHist", equal_hist_img)

    low_threshold = 100
    hight_threshold = 140
    canny_edge_img = cv2.Canny(equal_hist_img, low_threshold, hight_threshold, 3)
    cv2.imshow("Canny", canny_edge_img)

    corner = cv2.goodFeaturesToTrack(equal_hist_img, 1000, 0.01, 7)
    corner_points_img = canny_edge_img.copy()
    for c in corner:
        x, y = c.ravel()
        cv2.circle(corner_points_img, (x, y), 2, 255, 2)
    cv2.imshow("cornerPoints", corner_points_img)

    invert_img = cv2.bitwise_not(corner_points_img)
    distance_transform = cv2.distanceTransform(invert_img, cv2.DIST_L2, 3)
    distance_transformNorm = cv2.normalize(distance_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    cv2.imshow("distanceTransform", distance_transformNorm)

    integral_img = cv2.integral(gray_scale_img)
    height, width = gray_scale_img.shape
    mean_img = gray_scale_img.copy()
    k = 200
    for h in range (4, height-4):
        for w in range (4, width-4):
            eps = min(int(k * distance_transform[h, w]), 3)
            A = integral_img[h - eps - 1, w - eps - 1]
            B = integral_img[h + eps, w - eps - 1]
            C = integral_img[h - eps - 1, w + eps]
            D = integral_img[h + eps, w + eps]
            mean_img[h, w] = (A - B - C + D) / ((eps * 2 + 1)**2)
    cv2.imshow("meanFilt", mean_img)
    

def main():
    img = cv2.imread('home.jpg')
    if img is not None:
        cv2.imshow('Original image', img)
    elif img is None:
        print("Error loading image")
    filters(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
