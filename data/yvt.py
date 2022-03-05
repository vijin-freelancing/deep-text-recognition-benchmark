import os

import pandas as pd
import cv2
from tqdm import tqdm


if __name__ == '__main__':

    if __name__ == '__main__':
        frames_dir = '/home/vijin/career/freelancing/video-text-recognition/data/yvt/train/frames'
        output_dir = '/home/vijin/career/freelancing/video-text-recognition/data/yvt/train/words/frames'
        out_ann_file = '/home/vijin/career/freelancing/video-text-recognition/data/yvt/train/words/word-ann.csv'
        annotation_file = '/home/vijin/career/freelancing/video-text-recognition/data/yvt/train-ann.csv'

        os.makedirs(output_dir, exist_ok=True)
        annotation_df = pd.read_csv(annotation_file)

        # filter annotation file based on `lost`, `occluded`, `generated`
        annotation_df_filtered = annotation_df[(annotation_df['lost'] == 0) & (annotation_df['occluded'] == 0) &
                                               (annotation_df['generated'] == 0)]
        print(annotation_df.shape[0], annotation_df_filtered.shape[0])

        word_id = 1
        word_metdata = []
        for index, row in tqdm(annotation_df_filtered.iterrows()):
            video_id = row['video_id']
            frame_id = row['frame_id']
            transcription = row['label']
            xmin = row['xmin']
            ymin = row['ymin']
            xmax = row['xmax']
            ymax = row['ymax']

            image_path = os.path.join(frames_dir, f"{video_id}-{frame_id+1}.jpg")
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                word_image = image[ymin:ymax, xmin:xmax]

                word_file_name = os.path.join(output_dir, f"{video_id}-{frame_id}-{word_id}.jpg")
                cv2.imwrite(word_file_name, word_image)

                word_metdata.append({
                    'video_id': video_id,
                    'frame_id': frame_id,
                    'transcription': transcription,
                    'word_id': word_id
                })
            word_id += 1

        word_metadata_df = pd.DataFrame(word_metdata)
        word_metadata_df.to_csv(out_ann_file, index=False)
