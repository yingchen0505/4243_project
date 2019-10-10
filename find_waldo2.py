# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img_rgb = cv2.imread('datasets/JPEGImages/032.jpg')
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# template = cv2.imread('datasets/waldo.jpg',0)
# # saves the width and height of the template into 'w' and 'h'
# w, h = template.shape[::-1]
# res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
# threshold = 0.6
# # finding the values where it exceeds the threshold
# loc = np.where( res >= threshold)
# for pt in zip(*loc[::-1]):
#     #draw rectangle on places where it exceeds threshold
#     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)
# cv2.imwrite('found_waldo.png',img_rgb)

import numpy as np
import cv2
from matplotlib import pyplot as plt
import cyvlfeat

MIN_MATCH_COUNT = 10
test_image_ID = '042'

img1 = cv2.imread('datasets/waldo.jpg',0)          # queryImage
img2 = cv2.imread('datasets/JPEGImages/' + test_image_ID + '.jpg',0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3)
plt.savefig(test_image_ID + '_dsift_results.jpg', dpi=2000)

# Initiate SIFT detector
# sift = cv2.xfeatures2d.SIFT_create()
# sift = cv2.SIFT()
#
# # find the keypoints and descriptors with SIFT
# kp1, des1 = cyvlfeat.sift.sift(img1, compute_descriptor=True)
# kp2, des2 = cyvlfeat.sift.sift(img2, compute_descriptor=True)
# # # kp1, des1 = sift.detectAndCompute(img1,None)
# # # kp2, des2 = sift.detectAndCompute(img2,None)
# #
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 50)
#
# flann = cv2.FlannBasedMatcher(index_params, search_params)
#
# matches = flann.knnMatch(des1,des2,k=1)
# # matches = flann.knnMatch(des1,des2,k=2)
#
# # store all the good matches as per Lowe's ratio test.
# good = []
# for m,n in matches:
#     if m.distance < 0.7*n.distance:
#         good.append(m)
#
# if len(good)>MIN_MATCH_COUNT:
#     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#     matchesMask = mask.ravel().tolist()
#
#     h,w = img1.shape
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv2.perspectiveTransform(pts,M)
#
#     img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
#
# else:
#     print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
#     matchesMask = None
#
# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)
#
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
#
# plt.imshow(img3, 'gray')
# plt.savefig(test_image_ID + '_dsift_results.jpg', dpi=2000)
# # plt.savefig(test_image_ID + 'results.jpg', dpi=2000)
#
