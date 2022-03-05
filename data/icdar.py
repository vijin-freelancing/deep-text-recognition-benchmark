import os

import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


if __name__ == '__main__':
    frames_dir = '/home/vijin/career/freelancing/video-text-recognition/data/icdar-2015/train/frames'
    output_dir = '/home/vijin/career/freelancing/video-text-recognition/data/icdar-2015/train/words/frames'
    out_ann_file = '/home/vijin/career/freelancing/video-text-recognition/data/icdar-2015/train/words/word-ann.csv'
    annotation_file = '/home/vijin/career/freelancing/video-text-recognition/data/icdar-2015/train-ann.csv'

    os.makedirs(output_dir, exist_ok=True)
    annotation_df = pd.read_csv(annotation_file)

    word_id = 1
    word_metdata = []
    for index, row in tqdm(annotation_df.iterrows()):
        video_id = row['video_id']
        frame_id = row['frame_id']
        transcription = row['transcription']
        language = row['language']
        quality = row['quality']
        x1 = row['x1']
        x2 = row['x2']
        x3 = row['x3']
        x4 = row['x4']
        y1 = row['y1']
        y2 = row['y2']
        y3 = row['y3']
        y4 = row['y4']

        if not (str(transcription).startswith('#') or str(transcription).startswith('?')):
            polygon = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape((4, 2))
            image_path = os.path.join(frames_dir, f"{video_id}-{frame_id}.jpg")
            image = cv2.imread(image_path)
            warped = four_point_transform(image, polygon)
            word_file_name = os.path.join(output_dir, f"{video_id}-{frame_id}-{word_id}.jpg")
            cv2.imwrite(word_file_name, warped)

            word_metdata.append({
                'video_id': video_id,
                'frame_id': frame_id,
                'transcription': transcription,
                'language': language,
                'quality': quality,
                'word_id': word_id
            })

        word_id += 1
    word_metadata_df = pd.DataFrame(word_metdata)
    word_metadata_df.to_csv(out_ann_file, index=False)
