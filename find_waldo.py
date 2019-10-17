import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys, traceback
import os

template_path = 'datasets/Templates/'

# Read all templates
template_names = []
templates = []
# r=root, d=directories, f = files
for r, d, f in os.walk(template_path):
    for file in f:
        if '.jpg' in file:
            templates.append(os.path.join(r, file))
            template_names.append(str(file).strip('.jpg'))
        if '.png' in file:
            templates.append(os.path.join(r, file))
            template_names.append(str(file).strip('.png'))

MIN_MATCH_COUNT = 10

# Initialize output file
output_file = open("output/waldo.txt", "w+")

# Images IDs to run on
image_ids_file = open("datasets/ImageSets/train.txt", 'r')
image_ids = image_ids_file.readlines()

# Run on each image
for image_id in image_ids:
    image_id = image_id.strip('\n')
    print('datasets/JPEGImages/' + image_id + '.jpg')

    for index, template in enumerate(templates):
        img1 = cv2.imread(template, flags=cv2.IMREAD_GRAYSCALE)  # queryImage

        try:
            img2 = cv2.imread('datasets/JPEGImages/' + image_id + '.jpg', flags=cv2.IMREAD_GRAYSCALE)  # trainImage
            plt.imshow(img2)
            plt.show()
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
                plt.savefig('output/' + image_id + '_' + template_names[index] + '.jpg', dpi=2000)

                # Write output test file
                xmin = np.min(dst[:, :, 0])
                ymin = np.min(dst[:, :, 1])
                xmax = np.max(dst[:, :, 0])
                ymax = np.max(dst[:, :, 1])
                output_file.write(image_id + " 1.000 %.1f" % xmin + " %.1f" % ymin + " %.1f" % xmax + " %.1f" % ymax + " \n")

            else:
                print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
                matchesMask = None

        except:
            print(image_id + ' failed')
            traceback.print_exc(file=sys.stdout)

output_file.close()
