import math
import os
import random
import re
import sys

import cv2
import numpy as np


def contour_stuff(read_path):
    img = cv2.imread(read_path, 0)
    write_path = re.sub(".jpg", "_contour.jpg", read_path)
    temp_path = re.sub(".jpg", "_threshtemp.jpg", read_path)
    canny_path = re.sub(".jpg", "_canny.jpg", read_path)
    blur_path = re.sub(".jpg", "_blur.jpg", read_path)

    #blur = cv2.medianBlur(img, 81)
    blur = cv2.bilateralFilter(img,40,75,75)
    ret, thresh = cv2.threshold(blur, 120,255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(blur,0, 20)

    cv2.imwrite(temp_path, thresh)
    contours, hierarchy = cv2.findContours(edges, 1, 2)

    cnt = contours[0]
    epsilon = 100
    approx = cv2.approxPolyDP(cnt,epsilon,True)

    blah = np.array([[[0,0], [120,120], [0, 60], [200, 300]]])
    print blah
    print approx
    cv2.drawContours(img, approx, 0, (0,250,0), 20)


    cv2.imwrite(write_path, img)
    cv2.imwrite(canny_path, edges)
    cv2.imwrite(blur_path, blur)
    shape = edges.shape
    for i in range(100):
        pass
        #print edges[random.randint(0, shape[0]), random.randint(0, shape[1])]

def get_write_path(read_path, type_):
    write_path = read_path.replace("raw", type_)
    write_path = re.sub(".jpg", "_" + type_ + ".jpg", write_path)
    return write_path

def standardize_check(read_path):
    im = cv2.imread(read_path)

    blur_path = get_write_path(read_path, "blur")
    contour_path = get_write_path(read_path, "contour")
    thresh_path = get_write_path(read_path, "thresh")
    warp_path = get_write_path(read_path, "warp")
    thresh_warp_path = get_write_path(read_path, "thresh_warp")
    canny_path = get_write_path(read_path, "canny")
    denoise_path = get_write_path(read_path, "denoise")
    adaptive_path = get_write_path(read_path, "adaptive")

    # I do the writing to files right after the assignment, which is a little harder to organize,
    # but a lot of these functions modify things in place and it's hard to tell when they get 
    # touched.
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(1,1),1000)
    cv2.imwrite(blur_path, blur)

    thresh = apply_binary_threshold(gray)
    cv2.imwrite(thresh_path, thresh)

    #flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
    #thresh = apply_binary_threshold(blur)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    cv2.imwrite(canny_path, edged)

    kernel = np.ones((5,5),np.float32)/25
    smoothed = cv2.filter2D(edged,-1,kernel)
    cv2.imwrite(denoise_path, smoothed)


    contours, hierarchy = cv2.findContours(smoothed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea,reverse=True)[:2]

    check = contours[0]
    peri = cv2.arcLength(check, True)
    approx = cv2.approxPolyDP(check, 0.5 * peri, True)
    rect = cv2.minAreaRect(check)
    r = cv2.cv.BoxPoints(rect)
    corners = []
    for point in r:
        corners.append([int(y) for y in point])
    cv2.drawContours(im, np.array([corners]), 0, (0,250,0), 20)

    approx = approx.astype(np.float32)
    width = 2688
    height = 1520
    h = np.array([ [0,0],[width,0],[width,height],[0,height] ],np.float32)
    src = np.array(corners, dtype='float32')
    src = order_points(src)
    dst = h
    transform = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(im,transform,(width,height))
    warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(warp_path, warp)

    threshed_warp = apply_binary_threshold(warp)

    cv2.imwrite(contour_path, im)
    cv2.imwrite(thresh_warp_path, threshed_warp)

    adaptive_threshed = apply_adaptive_threshold(warp)
    cv2.imwrite(adaptive_path, adaptive_threshed)

def apply_binary_threshold(img):
    # Call something dark if it's darker than the 90th percentile pixel
    cutoff = np.percentile(img, 20)
    flag, threshed = cv2.threshold(img, 75, 255, cv2.THRESH_BINARY_INV)
    # The maxValue parameter is what you want to set things to when the threshold is exceeded
    # It is set to 0 (black) if not exceeded
    #threshed = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #    cv2.THRESH_BINARY,501, 2)
    return threshed

def apply_adaptive_threshold(img):
    # The maxValue parameter is what you want to set things to when the threshold is exceeded
    # It is set to 0 (black) if not exceeded
    threshed = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
        cv2.THRESH_BINARY,501, 2)
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

def process_all(path):
    files = [os.path.join(path, f) for f in os.listdir(path)]
    for file in files:
        if file.endswith(".jpg"):
            print "processing %s" % file
            standardize_check(file)

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

def get_pixel_chunk_sums(path, mapping):
    im = cv2.imread(path)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    sums = []
    for box in mapping:
        a, b, c, d = get_box_boundaries(box)
        chunk = gray[a:b, c:d]
        sums.append(chunk.sum())
    return np.array(sums, dtype="float32")

def get_pixel_chunk_sum_ratios(path_1, path_2, mapping):
    sums_1 = get_pixel_chunk_sums(path_1, mapping)
    sums_2 = get_pixel_chunk_sums(path_2, mapping)
    return sums_1 / sums_2

def find_abnormal_cells(path_1, path_2):
    im = cv2.imread(path_1)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    mapping = get_pixel_chunk_mapping(10, 32, gray)
    ratios = get_pixel_chunk_sum_ratios(path_1, path_2, mapping)
    weird = []
    for i in range(len(ratios)):
        ratio = ratios[i]
        if ratio < .5 or ratio > 1.8:
            print "Cell %s seems weird! Ratio of %s " % (i, ratio)
            weird.append(i)

    color_abnormalities(path_1, mapping, weird)
    return weird

def color_abnormalities(path, mapping, cell_nums):
    im = cv2.imread(path)
    for i in cell_nums:
        a, b, c, d = get_box_boundaries(mapping[i])
        # 0 means the blue index i supposed, because 10B 255R 255G is yellow, and this turns yellow
        # not sure how the ordering is known...
        im[a:b, c:d, 0] = 10
    abnormality_path = get_write_path(path, "abnormality")
    print abnormality_path
    cv2.imwrite(abnormality_path, im)



