# EDSNet: Efficient-DSNet for Video Summarization

## Getting Started

First, clone this project to your local environment.

```sh
git clone https://github.com/ashish2506prasad/EDSNet-Efficient-DSNet-for-Video-Summarization
```

Create a virtual environment with python.
```sh
conda create --name edsnet python=3.6
conda activate dsnet
```
or 
```sh
pip -m venv edsnet
edsnet/Scripts/activate
```
Install python dependencies.

```sh
pip install -r requirements.txt
```

## Datasets Preparation

Download the pre-processed datasets into `datasets/` folder, including [TVSum](https://github.com/yalesong/tvsum), [SumMe](https://gyglim.github.io/me/vsum/index.html), [OVP](https://sites.google.com/site/vsummsite/download), and [YouTube](https://sites.google.com/site/vsummsite/download) datasets.

If the Dropbox link is unavailable to you, try downloading from below links.

+ (kaggle) https://www.kaggle.com/datasets/meashish2003/vid-features/data

Now the datasets structure should look like

```
EDSNet
└── datasets/
    ├── eccv16_dataset_ovp_google_pool5.h5
    ├── eccv16_dataset_summe_google_pool5.h5
    ├── eccv16_dataset_tvsum_google_pool5.h5
    ├── eccv16_dataset_youtube_google_pool5.h5
    └── readme.txt
```
## Results
![image](https://github.com/user-attachments/assets/af83f2aa-285e-49af-a467-48ed3ee84499)

## Training

### Anchor-based

To train anchor-based attention model on TVSum and SumMe datasets with canonical settings, run

```sh
python train.py anchor-based --model-dir ../models/ab_basic --splits ../splits/tvsum.yml ../splits/summe.yml --base_model nystromformer --pooling_type roi --anchor_scales 12
```
To obtain the number of parameters run
```sh
python printing_model_information.py anchor-based --model-dir ../models/ab_basic --splits ../splits/tvsum.yml --base_model nystromformer --pooling_type roi --anchor_scales 12
```

To train on augmented and transfer datasets, run

```sh
python train.py anchor-based --model-dir ../models/ab_tvsum_aug/ --splits ../splits/tvsum_aug.yml --base_model nystromformer --pooling_type roi --anchor_scales 12
python train.py anchor-based --model-dir ../models/ab_tvsum_aug/ --splits ../splits/summe_aug.yml --base_model nystromformer --pooling_type roi --anchor_scales 12

python train.py anchor-based --model-dir ../models/ab_tvsum_aug/ --splits ../splits/tvsum_trans.yml --base_model nystromformer --pooling_type roi --anchor_scales 12
python train.py anchor-based --model-dir ../models/ab_tvsum_aug/ --splits ../splits/summe_trans.yml --base_model nystromformer --pooling_type roi --anchor_scales 12
```

To train with LSTM, Bi-LSTM or GCN feature extractor, specify the `--base-model` argument as `lstm`, `bilstm`, or `gcn`. For example,

```sh
python train.py anchor-based --model-dir ../models/ab_basic --splits ../splits/tvsum.yml ../splits/summe.yml --base-model lstm
```

for more training settings, refer `src/helpers/init_helper.py`

## Evaluation

To evaluate your anchor-based models, run

```sh
python evaluate.py anchor-based --model-dir ../models/ab_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml
```

## Generating Shots with KTS

Based on the public datasets provided by [DR-DSN](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce), we apply [KTS](https://github.com/pathak22/videoseg/tree/master/lib/kts) algorithm to generate video shots for OVP and YouTube datasets. Note that the pre-processed datasets already contain these video shots. To re-generate video shots, run

```sh
python make_shots.py --dataset ../datasets/eccv16_dataset_ovp_google_pool5.h5
python make_shots.py --dataset ../datasets/eccv16_dataset_youtube_google_pool5.h5
```

## Using Custom Videos

### Training & Validation

We provide scripts to pre-process custom video data, like the raw videos in `custom_data` folder.

First, create an h5 dataset. Here `--video-dir` contains several MP4 videos, and `--label-dir` contains ground truth user summaries for each video. The user summary of a video is a UxN binary matrix, where U denotes the number of annotators and N denotes the number of frames in the original video.

```sh
python make_dataset.py --video-dir ../custom_data/videos --label-dir ../custom_data/labels \
  --save-path ../custom_data/custom_dataset.h5 --sample-rate 15
```

Then split the dataset into training and validation sets and generate a split file to index them.

```sh
python make_split.py --dataset ../custom_data/custom_dataset.h5 \
  --train-ratio 0.67 --save-path ../custom_data/custom.yml
```

Now you may train on your custom videos using the split file.

```sh
python train.py anchor-based --model-dir ../models/custom --splits ../custom_data/custom.yml
python evaluate.py anchor-based --model-dir ../models/custom --splits ../custom_data/custom.yml
```

### Inference

To predict the summary of a raw video, use `infer.py`. For example, run

```sh
python infer.py anchor-based --ckpt-path ../models/custom/checkpoint/custom.yml.0.pt \
  --source ../custom_data/videos/EE-bNr36nyA.mp4 --save-path ./output.mp4
```

## Acknowledgments

We gratefully thank the below open-source repo, which greatly boost our research.

+ Thank [KTS](https://github.com/pathak22/videoseg/tree/master/lib/kts) for the effective shot generation algorithm.
+ Thank [DR-DSN](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce) for the pre-processed public datasets.
+ Thank [VASNet](https://github.com/ok1zjf/VASNet) and [DSNet](https://github.com/li-plus/DSNet) for the training and evaluation pipeline.

