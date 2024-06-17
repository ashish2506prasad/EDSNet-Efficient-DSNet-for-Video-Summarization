import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm
import os
from helpers import video_helper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', type=str, default='../custom_data/videos/')
    parser.add_argument('--label-dir', type=str, default='../custom_data/labels/')
    parser.add_argument('--sample-rate', type=int, default=15)
    parser.add_argument('--save-path', type=str, default='../custom_data/custom_dataset.h5')
    parser.add_argument('--feature-extractor', type=str, default='google-net',
                        choices=['google-net', 'swin-transformer', 'convnext'])
    parser.add_argument('--motion-feature', type=str, default=None)
    args = parser.parse_args()

    # create output directory
    out_dir = Path(args.save_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # annotation directory
    label_dir = Path(args.label_dir)
    

    # feature extractor
    print('Loading feature extractor ...')
    video_proc = video_helper.VideoPreprocessor(args.sample_rate, args.feature_extractor)

    # search all videos with .mp4 suffix
    # video_paths = sorted(Path(args.video_dir).glob('*.mp4'))

    video_list = sorted(os.listdir(args.video_dir))
    video_paths = [os.path.join(args.video_dir, video) for video in video_list if video.endswith('.mp4')]

    if args.motion_feature is not None:
        motion_feature_list = sorted(os.listdir(args.motion_feature))
        print(f'Loading motion features from {motion_feature_list} ...')   
        motion_features_paths = [os.path.join(args.motion_feature, motion_feature) for motion_feature in motion_feature_list if motion_feature.endswith('.npy')]

    print(f'Processing {len(video_paths)} videos ...')

    with h5py.File(args.save_path, 'w') as h5out:
        for idx, video_path in tqdm(list(enumerate(video_paths))):
            n_frames, features, cps, nfps, picks = video_proc.run(video_path)

            # load labels
            video_name = video_list[idx].split('.')[0]
            label_path = label_dir / f'{video_name}.json'
            with open(label_path) as f:
                data = json.load(f)
            user_summary = np.array(data['user_summary'], dtype=np.float32)
            _, label_n_frames = user_summary.shape
            # print video name and number of frames
            print(f'processing {video_name}: {n_frames} frames')
            # assert label_n_frames == n_frames, f'Invalid label of size {label_n_frames}: expected {n_frames}'

            try :
                assert n_frames == label_n_frames, f'Invalid label of size {user_summary.shape[1]}: expected {n_frames}'
            except AssertionError:
                print(f'Invalid label of size {user_summary.shape[1]}: expected {n_frames}')
                continue

            # compute ground truth frame scores
            gtscore = np.mean(user_summary[:, ::args.sample_rate], axis=0)

            # write dataset to h5 file
            video_key = f'{video_name}'
            h5out.create_dataset(f'{video_key}/features', data=features)
            h5out.create_dataset(f'{video_key}/gtscore', data=gtscore)
            h5out.create_dataset(f'{video_key}/user_summary', data=user_summary)
            h5out.create_dataset(f'{video_key}/change_points', data=cps)
            h5out.create_dataset(f'{video_key}/n_frame_per_seg', data=nfps)
            h5out.create_dataset(f'{video_key}/n_frames', data=n_frames)
            h5out.create_dataset(f'{video_key}/picks', data=picks)
            h5out.create_dataset(f'{video_key}/video_name', data=video_name)
            

            # load motin features
            if args.motion_feature is not None:
                motion_feature_path = motion_features_paths[idx]
                motion_features = np.load(motion_feature_path)
                motion_features = motion_features[::args.sample_rate, ::1]
                print(f'Loaded motion features of size {motion_features.shape}')
                # assert motion_features.shape[0] == n_frames, f'Invalid motion features of size {motion_features.shape[0]}: expected {n_frames}'
                h5out.create_dataset(f'{video_key}/motion_features', data=motion_features)

            

    print(f'Dataset saved to {args.save_path}')


if __name__ == '__main__':
    main()
