import copy
import json
import os
import random
import re
import sys

import cv2
import numpy as np
import scipy.stats


IMAGE_DIR = os.environ.get("FD_IMAGE_DIR")
NUM_ROWS = 30
NUM_COLS = 40
WIDTH_PIXELS = 2688
HEIGHT_PIXELS = 1520

def standardize_check(color, name):

    gray_path = IMAGE_DIR + "/gray/" + name
    contour_path = IMAGE_DIR + "/contour/" + name
    thresh_path = IMAGE_DIR + "/thresh/" + name
    warp_path = IMAGE_DIR + "/warp/" + name
    thresh_warp_path = IMAGE_DIR + "/thresh_warp/" + name
    canny_path = IMAGE_DIR + "/canny/" + name
    denoise_path = IMAGE_DIR + "/denoise/" + name
    adaptive_path = IMAGE_DIR + "/adaptive/" + name

    # I do the writing to files right after the assignment, which is a little harder to organize,
    # but a lot of these functions modify things in place and it's hard to tell when they get 
    # touched.
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    gray_2 = copy.deepcopy(gray)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    cv2.imwrite(gray_path, gray)

    thresh = apply_adaptive_threshold(gray)
    cv2.imwrite(thresh_path, thresh)

    edged = cv2.Canny(gray, 30, 200)
    cv2.imwrite(canny_path, edged)

    kernel = np.ones((20,20),np.float32)/400
    smoothed_canny = cv2.filter2D(edged, -1, kernel)
    cv2.imwrite(denoise_path, smoothed_canny)

    contours, hierarchy = cv2.findContours(smoothed_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea,reverse=True)[:2]

    check = contours[0]
    peri = cv2.arcLength(check, True)
    approx = cv2.approxPolyDP(check, 0.5 * peri, True)
    rect = cv2.minAreaRect(check)
    r = cv2.cv.BoxPoints(rect)
    corners = []
    for point in r:
        corners.append([int(c) for c in point])
    cv2.drawContours(color, np.array([corners]), 0, (0,250,0), 20)

    approx = approx.astype(np.float32)
    width = HEIGHT_PIXELS
    height = WIDTH_PIXELS
    src = np.array(corners, dtype='float32')
    src = order_points(src)
    dst = np.array([ [0,0],[width,0],[width,height],[0,height] ],np.float32)
    transform = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(gray_2, transform, (width,height))
    cv2.imwrite(warp_path, warp)

    cv2.imwrite(contour_path, color)

    adaptive_threshed = apply_binary_threshold(warp)
    cv2.imwrite(adaptive_path, adaptive_threshed)

def apply_inv_binary_threshold(img):
    flag, threshed = cv2.threshold(img, 75, 255, cv2.THRESH_BINARY_INV)
    return threshed

def apply_binary_threshold(img):
    flag, threshed = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    return threshed

def apply_adaptive_threshold(img):
    # The maxValue parameter is what you want to set things to when the threshold is exceeded
    # It is set to 0 (black) if not exceeded
    threshed = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
        cv2.THRESH_BINARY, 11, 3)
    return threshed

def order_points(pts):
    # Borrowed from some really helpful dude on the web
    # TODO cite this

    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
 
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
 
    # return the ordered coordinates
    return rect


def get_color_img(path):
    color = cv2.imread(path)
    assert color is not None
    return color

def get_gray_img(path):
    color = get_color_img(path)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    assert gray is not None
    return gray

def get_adaptive_gray_img(name):
    adaptive_path = IMAGE_DIR + "/warp/" + name
    gray = get_gray_img(adaptive_path)
    return gray

def process_all(path):
    full_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]
    for i in range(len(full_paths)):
        print "processing %s" % full_paths[i]
        color = get_color_img(full_paths[i])
        standardize_check(color=color, name=get_basename(full_paths[i]))

def get_pixel_chunk_mapping(num_rows, num_columns, gray):
    pixels_per_row = int(gray.shape[0] / num_rows)
    pixels_per_column = int(gray.shape[1] / num_columns)
    coords = []
    for i in range(num_rows):
        start_vert = i * pixels_per_row
        end_vert = start_vert + pixels_per_row
        #print "vert: %s %s" % (start_vert, end_vert)
        for j in range(num_columns):
            start_horiz = j * pixels_per_column
            end_horiz = start_horiz + pixels_per_column
            #print "horiz: %s %s" % (start_horiz, end_horiz)
            coords.append({
                "start_vert": start_vert,
                "end_vert": end_vert,
                "start_horiz": start_horiz,
                "end_horiz": end_horiz
            })
    return coords

def get_box_boundaries(box):
    start_vert = box["start_vert"]
    end_vert = box["end_vert"]
    start_horiz = box["start_horiz"]
    end_horiz = box["end_horiz"]
    return start_vert, end_vert, start_horiz, end_horiz

def get_pixel_chunk_sums(gray, mapping):
    sums = []
    for box in mapping:
        a, b, c, d = get_box_boundaries(box)
        chunk = gray[a:b, c:d]
        stat = int((chunk == 0).sum() > 200)
        sums.append(stat)
    return np.array(sums, dtype="float32")

def find_abnormal_cells(gray, reference_dists):
    mapping = get_pixel_chunk_mapping(NUM_ROWS, NUM_COLS, gray)
    sums = get_pixel_chunk_sums(gray, mapping)
    weird = []
    for i, chunk_sum in enumerate(sums):
        # Woops, apparently you can have ints as keys for dicts, but when you JSON serialize they 
        # become strings?  Need to look into that...
        ref_dist = reference_dists[str(i)]
        mean = ref_dist["mean"]
        if mean > .9 and chunk_sum < 1:
            print "Chunk %s seems weird!" % i
            weird.append(i)
    return weird

def color_abnormalities(name, color, mapping, cell_nums):
    for i in cell_nums:
        a, b, c, d = get_box_boundaries(mapping[i])
        # 0 means the blue index i supposed, because 10B 255R 255G is yellow, and this turns yellow
        # not sure how the ordering is known...
        color[a:b, c:d, 0] = 10
        color[a:b, c:d, 1] = 255
        color[a:b, c:d, 2] = 255
    abnormality_path = IMAGE_DIR + "/abnormality/" + name
    cv2.imwrite(abnormality_path, color)

def make_reference_dists(path):
    files = [f for f in os.listdir(path) if f.endswith(".jpg")]
    stats_by_chunk = {}
    mapping = None
    for rep_ind in range(250):
        f = random.choice(files)
        print "processing %s" % f
        adaptive_path = IMAGE_DIR + "/adaptive/" + f
        adaptive = get_gray_img(adaptive_path)
        if mapping is None:
            assert isinstance(adaptive, np.ndarray)
            mapping = get_pixel_chunk_mapping(NUM_ROWS, NUM_COLS, adaptive)
        chunk_sums = get_pixel_chunk_sums(adaptive, mapping)
        for chunk_ind, chunk_sum in enumerate(chunk_sums):
            if chunk_ind not in stats_by_chunk:
                stats_by_chunk[chunk_ind] = [chunk_sum]
            else:
                stats_by_chunk[chunk_ind].append(chunk_sum)

    dist_by_chunk = {}
    for chunk_ind, chunk in stats_by_chunk.iteritems():
        # Apparently np numeric types arent json serializable?
        dist_by_chunk[chunk_ind] = {"mean": float(np.mean(chunk))}
    with open("reference_dists.json", "w") as f:
        f.write(json.dumps(dist_by_chunk))

def get_reference_dists_from_file():
    with open("reference_dists.json", "r") as f:
        return json.loads(f.read())

def get_basename(path):
    return os.path.basename(path)

def scan_image_for_fraud(path):
    color = get_color_img(path)
    name = get_basename(path)
    standardize_check(color, get_basename(path))
    ref_dists = get_reference_dists_from_file()
    adaptive_path = IMAGE_DIR + "/adaptive/" + name
    adaptive = get_gray_img(adaptive_path)
    abnormal_cells = find_abnormal_cells(adaptive, ref_dists)
    name = get_basename(path)
    mapping = get_pixel_chunk_mapping(NUM_ROWS, NUM_COLS, adaptive)
    # The color object gets modified in the other functions, so we need to make a copy
    color_for_abnorm_highlight = get_color_img(adaptive_path)
    color_abnormalities(name, color_for_abnorm_highlight, mapping, abnormal_cells)
    visualize_ref_dists(ref_dists, mapping)

def visualize_ref_dists(ref_dists, mapping):
    img = np.zeros((HEIGHT_PIXELS, WIDTH_PIXELS), dtype=np.int) + 255
    for chunk_ind, chunk_dist in ref_dists.iteritems():
        a, b, c, d = get_box_boundaries(mapping[int(chunk_ind)])
        img[a:b, c:d] = 255 - int(chunk_dist["mean"] * 255)
    cv2.imwrite(IMAGE_DIR + "/misc/ref_dists_visual.jpg", img)


