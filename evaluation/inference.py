import os
import string

import torch
from infer_utils import ViTSTRFeatureExtractor
import pandas as pd
from tqdm import tqdm


class TokenLabelConverter:
    """ Convert between text-label and text-index """

    def __init__(self, character, batch_max_length, device):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        self.SPACE = '[s]'
        self.GO = '[GO]'
        self.list_token = [self.GO, self.SPACE]
        self.character = self.list_token + list(character)
        self.device = device

        self.dict = {word: i for i, word in enumerate(self.character)}
        self.batch_max_length = batch_max_length + len(self.list_token)

    def encode(self, text):
        """ convert text-label into text-index.
        """
        length = [len(s) + len(self.list_token) for s in text]  # +2 for [GO] and [s] at end of sentence.
        batch_text = torch.LongTensor(len(text), self.batch_max_length).fill_(self.dict[self.GO])
        for i, t in enumerate(text):
            txt = [self.GO] + list(t) + [self.SPACE]
            txt = [self.dict[char] for char in txt]
            batch_text[i][:len(txt)] = torch.LongTensor(txt)  # batch_text[:, 0] = [GO] token
        return batch_text.to(self.device)

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


def infer(image_path, converter, model):
    img = ViTSTRFeatureExtractor()(image_path)
    with torch.no_grad():
        pred = model(img, seqlen=converter.batch_max_length)
        _, pred_index = pred.topk(1, dim=-1, largest=True, sorted=True)
        pred_index = pred_index.view(-1, converter.batch_max_length)
        length_for_pred = torch.IntTensor([converter.batch_max_length - 1] )
        pred_str = converter.decode(pred_index[:, 1:], length_for_pred)
        pred_EOS = pred_str[0].find('[s]')
        pred_str = pred_str[0][:pred_EOS]
    return pred_str


if __name__ == '__main__':
    character = string.printable[:-6]
    input_channel = 1
    model_path = '/home/vijin/career/freelancing/github/deep-text-recognition-benchmark/weights/vitstr_small_patch16_224_aug_infer.pth'
    # model_path = '/home/vijin/career/freelancing/github/deep-text-recognition-benchmark/weights/vitstr_tiny_patch16_224_aug.pth'
    batch_max_length = 25

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    converter = TokenLabelConverter(character=character, batch_max_length=batch_max_length, device=device)

    # load model
    model = torch.load(model_path, map_location=device)
    model.eval()

    # annotation_file = '/home/vijin/career/freelancing/video-text-recognition/data/icdar-2013/test/words/word-ann.csv'
    # frames_dir = '/home/vijin/career/freelancing/video-text-recognition/data/icdar-2013/test/words/frames'

    annotation_file = '/home/vijin/career/freelancing/video-text-recognition/data/yvt/test/words/word-ann.csv'
    frames_dir = '/home/vijin/career/freelancing/video-text-recognition/data/yvt/test/words/frames'

    ann_df = pd.read_csv(annotation_file)

    # CAitwP5Za_I_10, c4BLVznuWnU_13 has annotation issues
    ann_df = ann_df[
        ~ann_df.video_id.isin(['CAitwP5Za_I_10', 'c4BLVznuWnU_13'])]

    pred_data = []
    for _, row in tqdm(ann_df.iterrows()):
        image_path = os.path.join(frames_dir, f"{row['video_id']}-{row['frame_id']}-{row['word_id']}.jpg")
        row_dict = row.to_dict()
        pred = infer(image_path, converter, model)
        if pred:
            row_dict['pred'] = pred
        pred_data.append(row_dict)

    pred_df = pd.DataFrame(pred_data)
    # pred_df.to_csv(os.path.join('/home/vijin/career/freelancing/video-text-recognition/recognition/ViTSTR/vitstr_small_patch16_224_aug_icdar_2013_test.csv'), index=False)
    pred_df.to_csv(os.path.join(
        '/home/vijin/career/freelancing/video-text-recognition/recognition/ViTSTR/vitstr_small_patch16_224_aug_yvt_test.csv'),
                   index=False)


    # pred_df = pd.read_csv('/home/vijin/career/freelancing/video-text-recognition/recognition/ViTSTR/vitstr_small_patch16_224_aug_icdar_2013_test.csv')
    pred_df['match'] = pred_df.apply(lambda row: 1 if str(row['pred']).lower() == str(row['transcription']).lower() else 0, axis=1)

    # low_quality = pred_df[pred_df['quality'] == 'LOW']
    # high_quality = pred_df[pred_df['quality'] == 'HIGH']
    # medium_quality = pred_df[pred_df['quality'] == 'MODERATE']
    # high_medium_quality = pred_df[~(pred_df['quality'] == 'LOW')]

    print(pred_df.match.value_counts(normalize=True) * 100)
    # print(high_quality.match.value_counts(normalize=True) * 100)
    # print(medium_quality.match.value_counts(normalize=True) * 100)
    # print(high_medium_quality.match.value_counts(normalize=True) * 100)
    # print(low_quality.match.value_counts(normalize=True) * 100)
