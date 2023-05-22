# AICUP_audio_2023
此repository為AICUP-2023 多模態病理嗓音分類競賽，用以上傳程式以及提交的答案用的。

# Leaderboard
|Public Score|Public Rank|Private Score|Private Rank|
|-|-|-|-|
|0.600654|7/371|0.607568|6/371|



# Getting the code
Git link

# Repository structure
```
還沒寫
```

# Setting the environment
```
conda create -n pytorch-gpu python=3.9
conda activate pytorch-gpu
```
根據合適的顯卡版本安裝[pytorch、cudatoolkit](https://pytorch.org/get-started/previous-versions/)
```
# CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
安裝必須庫
```
pip install librosa
pip install sklearn
```
若因模組更新導致出現問題，可指定版本
```
pip install librosa==0.10.0.post2
pip install sklearn==1.2.2
```

# Prepared dataset
[Google Drive Link](https://drive.google.com/drive/folders/10YqPS2SABOZw6mT9jD5gEUhVc2MnXaMK?usp=sharing)
```
MFCC製作過程
```

# Download the best model
在[repository](https://github.com/JulianLee310514065/AICUP_audio_2023/#Repository-structure)中有五個.pth檔，即為最好的模型，下載使用即可。

# Training
```
介紹
```


# Inference (public、private data)
```
介紹
```

# Reproducing submission
若要重現最終提交結果，可以做以下步驟，最後上傳結果也有在[repository](https://github.com/JulianLee310514065/AICUP_audio_2023/#Repository-structure)中，為`submission.csv`

# Acknowledgement
我們參考了以下網站

# Citation
```
@misc{
    title  = {AICUP_audio_2023},
    author = {Chang-Yi Lee}
    url    = {https://github.com/JulianLee310514065/AICUP_audio_2023},
    year   = {2023}
}
```
