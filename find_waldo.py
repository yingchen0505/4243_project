import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

output_file = open("output/waldo.txt", "w+")

# template_path = 'datasets/{}.jpg'

img1 = cv2.imread('datasets/Templates/waldo.jpg', 0)  # queryImage

test_ids_file = open("datasets/ImageSets/val.txt", 'r')
test_ids = test_ids_file.readlines()

for test_image_ID in test_ids:
    test_image_ID = test_image_ID.strip('\n')
    print('datasets/JPEGImages/' + test_image_ID + '.jpg')
    img2 = cv2.imread('datasets/JPEGImages/' + test_image_ID + '.jpg', 0)  # trainImage
    # plt.imshow(img2)
    # plt.show()
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

        h, w = img1.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        dst = cv2.perspectiveTransform(pts, M)
        dst += (w, 0)  # adding offset

        img3 = cv2.polylines(img3, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

        plt.imshow(img3)
        plt.savefig('output/' + test_image_ID + '_sift_results_ransac_boundingbox.jpg', dpi=2000)

        # Write output test file
        xmin = np.min(dst[:, :, 0])
        ymin = np.min(dst[:, :, 1])
        xmax = np.max(dst[:, :, 0])
        ymax = np.max(dst[:, :, 1])
        output_file.write(test_image_ID + " 1.000 %.1f" % xmin + " %.1f" % ymin + " %.1f" % xmax + " %.1f" % ymax + " \n")

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

output_file.close()
