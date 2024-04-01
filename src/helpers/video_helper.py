from os import PathLike
from pathlib import Path
import logging
import cv2
import numpy as np
import torch
from PIL import Image
from numpy import linalg
from torch import nn
from torchvision import transforms, models
import init_helper

from kts.cpd_auto import cpd_auto
import os

logger = logging.getLogger()

def ends_with_mp4(path):
    _, extension = os.path.splitext(path)
    return extension.lower() == '.mp4' 


class FeatureExtractor(object):
    def __init__(args, self):
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        model_google_net = models.googlenet(pretrained=True)
        model_google_net = nn.Sequential(*list(model_google_net.children())[:-2])
        model_google_net = model_google_net.cuda().eval()

        model_convnext = models.convnext_large(pretrained=True)
        model_convnext = nn.Sequential(*list(model_convnext.children())[:-1])
        model_convnext = model_convnext.cuda().eval()

        model_swin_b = models.swin_v2_b(pretrained=True)
        model_swin_b = nn.Sequential(*list(model_swin_b.children())[:-1])
        model_swin_b = model_swin_b.cuda().eval()

        if args.feature_extractor == 'google-net':
            self.model = model_google_net
        elif args.feature_extractor =='swin-transformer':
            self.model = model_swin_b
        elif args.feature_extractor == 'convnext':
            self.model = model_convnext


    def run(self, img: np.ndarray) -> np.ndarray:
        img = Image.fromarray(img)
        img = self.preprocess(img)
        batch = img.unsqueeze(0)
        with torch.no_grad():
            feat = self.model(batch.cuda())
            feat = feat.view(-1,1).cpu().numpy()

        assert feat.shape == (1024,), f'Invalid feature shape {feat.shape}: expected 1024'
        # normalize frame features
        feat /= linalg.norm(feat) + 1e-10
        return feat


class VideoPreprocessor(object):
    def __init__(self, sample_rate: int) -> None:
        self.model = FeatureExtractor()
        self.sample_rate = sample_rate

    def get_features(self, video_path: PathLike):
        if ends_with_mp4(video_path):
            video_path = Path(video_path)
            cap = cv2.VideoCapture(str(video_path))
            
            assert cap is not None, f'Cannot open video: {video_path}'

            features = []
            n_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if n_frames % self.sample_rate == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    feat = self.model.run(frame)
                    features.append(feat)

                n_frames += 1

            cap.release()

            features = np.array(features)
            return n_frames, features
        

    def kts(self, n_frames, features):
        seq_len = len(features)
        picks = np.arange(0, seq_len) * self.sample_rate

        # compute change points using KTS
        kernel = np.matmul(features, features.T)
        change_points, _ = cpd_auto(kernel, seq_len - 1, 1, verbose=False)
        change_points *= self.sample_rate
        change_points = np.hstack((0, change_points, n_frames))
        begin_frames = change_points[:-1]
        end_frames = change_points[1:]
        change_points = np.vstack((begin_frames, end_frames - 1)).T

        n_frame_per_seg = end_frames - begin_frames
        return change_points, n_frame_per_seg, picks

    def run(self, video_path: PathLike):
        n_frames, features = self.get_features(video_path)
        cps, nfps, picks = self.kts(n_frames, features)
        return n_frames, features, cps, nfps, picks
    

if __name__ == '__main__':
    pass
