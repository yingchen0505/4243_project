import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys, traceback
import os

template_path = 'datasets/crop_train_nonsimilar/'

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
output_file_waldo = open("output/waldo.txt", "w+")
output_file_wenda = open("output/wenda.txt", "w+")
output_file_wizard = open("output/wizard.txt", "w+")

# Images IDs to run on
image_ids_file = open("datasets/ImageSets/val.txt", 'r')
# image_ids_file = open("datasets/ImageSets/train.txt", 'r')
image_ids = image_ids_file.readlines()

template_sifts = []

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# counter = 0
# Build sift for all the templates
for index, template in enumerate(templates):
    print(template_names[index])
    img1 = cv2.imread(template, flags=cv2.IMREAD_GRAYSCALE)  # queryImage
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    template_sifts.append({'kp1': kp1, 'des1': des1, 'img1': img1, 'template_name': template_names[index]})

    # counter += 1
    # if counter > 5:
    #     break


# find center of gravity for four points
# if the diagonals are of similar lengths, it is a legit box
def is_rectangle(x1, y1, x2, y2, x3, y3, x4, y4):
    cx = (x1 + x2 + x3 + x4) / 4
    cy = (y1 + y2 + y3 + y4) / 4

    dd1 = np.square(cx - x1) + np.square(cy - y1)
    dd2 = np.square(cx - x2) + np.square(cy - y2)
    dd3 = np.square(cx - x3) + np.square(cy - y3)
    dd4 = np.square(cx - x4) + np.square(cy - y4)

    threshold = 0.5
    diagonals = np.array((dd1, dd2, dd3, dd4))
    if np.std(diagonals) / np.mean(diagonals) < threshold:
        return True
    else:
        return False


def polygon_area(corners):
    n = len(corners)  # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


# if the bounding box is insanely small, it's not a legit box
def is_too_small(pts, img_size):
    threshold = 0.00001
    box_area = polygon_area(pts)
    if box_area / (img_size[0] * img_size[1]) < threshold:
        return True
    else:
        return False


# Return true if any point falls outside of image
def points_out_of_bound(pts, img_size):
    if np.any(pts < 0):
        return True
    # image size in (y, x)
    # points in (x, y)
    if np.any(pts[:, 0] >= img_size[1]):
        return True

    if np.any(pts[:, 1] >= img_size[0]):
        return True

    return False


# Run on each image
for image_id in image_ids:
    image_id = image_id.strip('\n')
    print('datasets/JPEGImages/' + image_id + '.jpg')

    try:
        img2 = cv2.imread('datasets/JPEGImages/' + image_id + '.jpg', flags=cv2.IMREAD_GRAYSCALE)  # trainImage
        scaling_factor = np.ceil(max(img2.shape) / 1600)
        print(img2.shape)
        print(scaling_factor)
        new_size = np.array(np.rint(img2.shape / scaling_factor), dtype=int)
        new_size = np.array((new_size[1], new_size[0]))
        new_size = tuple(new_size)
        print(new_size)
        img2_resized = cv2.resize(img2, new_size)
        # plt.imshow(img2_resized)
        # plt.show()
        kp2, des2 = sift.detectAndCompute(img2_resized, None)

        for index, template_sift in enumerate(template_sifts):
            kp1 = template_sift['kp1']
            des1 = template_sift['des1']
            img1 = template_sift['img1']

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
                img2_resized = cv2.polylines(img2_resized, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
                draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                   singlePointColor=None,
                                   matchesMask=matchesMask,  # draw only inliers
                                   flags=2)

                # cv2.drawMatchesKnn expects list of lists as matches.
                img3 = cv2.drawMatches(img1, kp1, img2_resized, kp2, good, None, **draw_params)

                h, w = img1.shape[:2]
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

                dst = cv2.perspectiveTransform(pts, M)
                dst += (w, 0)  # adding offset

                if points_out_of_bound(dst[:, 0, :], img2_resized.shape):
                    print("Out of Bound: " + str(dst))
                    continue

                if is_too_small(dst[:, 0, :], img2_resized.shape):
                    print("Too small: " + str(dst))
                    continue

                if not is_rectangle(
                        dst[0][0][0], dst[0][0][1], dst[1][0][0], dst[1][0][1], dst[2][0][0], dst[2][0][1],
                        dst[3][0][0], dst[3][0][1]):
                    print("Bad bounding box: " + str(dst))
                    continue

                img3 = cv2.polylines(img3, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

                # plt.imshow(img3)
                # plt.savefig('output/' + image_id + '_' + template_names[index] + '.jpg', dpi=2000)
                cv2.imwrite('output/' + image_id + '_' + template_names[index] + '.jpg', img3)

                # remove offset
                dst[:, :, 0] -= w
                # Scale back
                dst *= scaling_factor

                # Write output test file
                xmin = np.min(dst[:, :, 0])
                ymin = np.min(dst[:, :, 1])
                xmax = np.max(dst[:, :, 0])
                ymax = np.max(dst[:, :, 1])
                output_string = \
                    image_id + " 1.000 %.1f" % xmin + " %.1f" % ymin + " %.1f" % xmax + " %.1f" % ymax + " \n"

                if "waldo" in template_sift['template_name']:
                    output_file_waldo.write(output_string)

                elif "wenda" in template_sift['template_name']:
                    output_file_wenda.write(output_string)

                elif "wizard" in template_sift['template_name']:
                    output_file_wizard.write(output_string)

            else:
                print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
                matchesMask = None

    except:
        print(image_id + ' failed')
        traceback.print_exc(file=sys.stdout)

output_file_waldo.close()
output_file_wenda.close()
output_file_wizard.close()
