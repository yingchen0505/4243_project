import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys, traceback
import os

template_path = 'datasets/crop_train/'
output_path = 'output1/'

# Read all templates
# These templates are pre-cropped from the training set 
# using another set of code in the 'crop.ipynb' notebook
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

# The minimum number of matches needed for the algorithm to proceed to draw bounding boxes
# In our case, the best param setting is found to be 0, since we want to be as lenient as possible
# To obtain more possible waldos
MIN_MATCH_COUNT = 0

# Initialize output files
output_file_waldo = open(output_path + "waldo.txt", "w+")
output_file_wenda = open(output_path + "wenda.txt", "w+")
output_file_wizard = open(output_path + "wizard.txt", "w+")

# Images IDs to run on
# These are the test images we want to see how well how templates can match with
image_ids_file = open("datasets/ImageSets/val.txt", 'r')
image_ids = image_ids_file.readlines()

# A list to store the sift key points and descriptors extracted from the templates
# Since it takes a long time to extract the descriptors
template_sifts = []

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()


# This is a helper function to generate the new size when we perform 
# Downsizing of an image
def get_new_size(img_shape, down_scaling_factor):
    size = np.array(np.rint(np.array(img_shape, dtype=int) / down_scaling_factor), dtype=int)
    size = np.array((size[1], size[0]))
    return tuple(size)


# Calculate sift key point and descriptors for all the templates
for index, template in enumerate(templates):
    print(template_names[index])
    img1 = cv2.imread(template, flags=cv2.IMREAD_GRAYSCALE)  # Template
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    template_sifts.append({'kp1': kp1, 'des1': des1, 'img1': img1, 'template_name': template_names[index]})

	# Augment the template set by
    # Downsizing bigger templates
	# To generate additional templates
    if max(img1.shape) > 100:
        print('downsizing ' + template_names[index])
        new_size = get_new_size(img1.shape, 2)
		# Only stop if the resized new template has edges shorter than 10 pixels
        while np.max(new_size) > 10:
            img_resized = cv2.resize(img1, new_size)
            kp1, des1 = sift.detectAndCompute(img_resized, None)
            template_sifts.append({
                'kp1': kp1, 'des1': des1, 'img1': img_resized,
                'template_name': template_names[index] + '_size' + str(new_size)})
            new_size = get_new_size(img_resized.shape, 2)


# find center of gravity for four points
# if the diagonals are of similar lengths, it is a legit box
# if the four edges are of similar lengths, it is a legit box
def is_rectangle(x1, y1, x2, y2, x3, y3, x4, y4):
	# Calculate center of gravity
    cx = (x1 + x2 + x3 + x4) / 4
    cy = (y1 + y2 + y3 + y4) / 4

	# Calculate the four "diagonals", 
	# i.e. distance from CG to the four corners
    dd1 = np.square(cx - x1) + np.square(cy - y1)
    dd2 = np.square(cx - x2) + np.square(cy - y2)
    dd3 = np.square(cx - x3) + np.square(cy - y3)
    dd4 = np.square(cx - x4) + np.square(cy - y4)

    diagnonal_threshold = 0.1
    diagonals = np.array((dd1, dd2, dd3, dd4))
    diagonals_legit_by_std = np.std(diagonals) / np.mean(diagonals) < diagnonal_threshold
    diagnonal_ratio_threshold = 0.1
    diagonals_legit = np.min(diagonals) / np.max(diagonals) > diagnonal_ratio_threshold

    edge1 = np.square(x2 - x1) + np.square(y2 - y1)
    edge2 = np.square(x3 - x2) + np.square(y3 - y2)
    edge3 = np.square(x4 - x3) + np.square(y4 - y3)
    edge4 = np.square(x1 - x4) + np.square(y1 - y4)
    edges = np.array((edge1, edge2, edge3, edge4))
    edges_ratio_threshold = 0.1
    edges_legit = np.min(edges) / np.max(edges) > edges_ratio_threshold

    if diagonals_legit_by_std & diagonals_legit & edges_legit:
        return True
    else:
        return False


# This is a helper function that calculates the area of a polygon
# Formed by the corners using the shoelace formula
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
def is_too_small(pts):
    box_area = polygon_area(pts)
    if box_area < 100:
        return True
    else:
        return False


# If the points in normal order of 1,2,3,4 has smaller area than 1,2,4,3,
# The points form a twisted rectangle and therefore the box is not legit
def is_twisted(pts):
    normal_area = polygon_area(pts)
	# Check for all permutations by taking each of the four points 
	# as the "start" point
	# Check of swapping the last two points generates a larger area than
	# the normal order of points
    for i in range(len(pts)):
        twisted_points = np.array((pts[i], pts[(i+1) % 4], pts[(i+3) % 4], pts[(i+2) % 4]))
        twisted_area = polygon_area(twisted_points)
        if normal_area < twisted_area:
            return True
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


# Run on each test image to find matches with templates, and draw bounding boxes
for image_id in image_ids:
    image_id = image_id.strip('\n')
    print('datasets/JPEGImages/' + image_id + '.jpg')

    try:
        img2 = cv2.imread('datasets/JPEGImages/' + image_id + '.jpg', flags=cv2.IMREAD_GRAYSCALE)  # validation image
		
		# Downsizing larger test images since OpenCV has a size limit for SIFT
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
		
		# Calculate keypoints and descriptors for the test image
        kp2, des2 = sift.detectAndCompute(img2_resized, None)

		# Try to match with each template
        for index, template_sift in enumerate(template_sifts):
            kp1 = template_sift['kp1']
            des1 = template_sift['des1']
            img1 = template_sift['img1']

			# NN matching
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            # Apply ratio test
			# Actually, empirically we do better without the ratio test!
            good = []
            for m, n in matches:
                # if m.distance < 0.85 * n.distance:
                good.append(m)

			# Try to draw bounding boxes
            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
				# RANSAC
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()
                h, w = img1.shape[:2]
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

                try:
                    dst = cv2.perspectiveTransform(pts, M)

                except:
                    print(template_sift['template_name'] + ' failed')
                    traceback.print_exc(file=sys.stdout)
                    continue

                if points_out_of_bound(dst[:, 0, :], img2_resized.shape):
                    print("Out of Bound: " + str(dst))
                    continue

                if is_twisted(dst[:, 0, :]):
                    print("Twisted: " + str(dst))
                    continue

                if is_too_small(dst[:, 0, :]):
                    print("Too small: " + str(dst))
                    continue

                if not is_rectangle(
                        dst[0][0][0], dst[0][0][1], dst[1][0][0], dst[1][0][1], dst[2][0][0], dst[2][0][1],
                        dst[3][0][0], dst[3][0][1]):
                    print("Bad bounding box: " + str(dst))
                    continue

                dst += (w, 0)  # adding offset
                draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                   singlePointColor=None,
                                   matchesMask=matchesMask,  # draw only inliers
                                   flags=2)

                # cv2.drawMatchesKnn expects list of lists as matches.
                img3 = cv2.drawMatches(img1, kp1, img2_resized, kp2, good, None, **draw_params)
                img3 = cv2.polylines(img3, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

                # plt.imshow(img3)
                # plt.savefig(output_path + image_id + '_' + template_sift['template_name'] + '.jpg', dpi=2000)
                cv2.imwrite(output_path + image_id + '_' + template_sift['template_name'] + '.jpg', img3)

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