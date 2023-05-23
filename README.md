# AICUP_audio_2023
此repository為AICUP-2023 多模態病理嗓音分類競賽，用以上傳程式以及提交的答案用的。

# Leaderboard
|Public Score|Public Rank|Private Score|Private Rank|
|-|-|-|-|
|0.600654|7 / 371|0.607568|6 / 371|



# Getting the code
Git link

# Repository structure
```
┌ submit┌ output_mfcc13.npy
│       ├ output_mfcc17.npy
│       ├ output_mfcc21.npy
│       ├ output_mfcc30.npy
│       └ output_mfcc50.npy
│
├ 1_DataPreprocessing.ipynb
├ 2_AI_CUP_mfcc13.ipynb
├ 2_AI_CUP_mfcc17.ipynb
├ 2_AI_CUP_mfcc21.ipynb
├ 2_AI_CUP_mfcc30.ipynb
├ 2_AI_CUP_mfcc50.ipynb
├ 3_Ensemble_pub_pri.ipynb
├ LICENSE
├ README.md
├ mfcc13_use_all.pth
├ mfcc17_use_all.pth
├ mfcc21_use_all.pth
├ mfcc30_use_all.pth
├ mfcc50_use_all.pth
└ submission.csv

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
> 訓練資料連結
[Google Drive Link](https://drive.google.com/drive/folders/10YqPS2SABOZw6mT9jD5gEUhVc2MnXaMK?usp=sharing)

於`1_DataPreprocessing.ipynb`有定義`make_mfcc`函數，可透過輸入`DataFrame`與`n_mfcc`來獲得可供訓練或驗證的`.npy`檔。
```python
def make_mfcc(df:pd.DataFrame, n_mfcc=13):
    for file_path in df['wave_path'].to_list():

        signal_tem, sample_rate = librosa.load(file_path, sr=44100)
        signal = signal_tem[:44100]        

        n_fft = int(16/1000 * sample_rate)  
        hop_length = int(8/1000 * sample_rate)

        # MFCCs
        MFCCs = librosa.feature.mfcc(y=signal, sr =sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
        # print(MFCCs.shape)

        np.save(file_path.replace('.wav', f'_mfcc_{n_mfcc}.npy'), MFCCs)
```


# Download the best model
在[repository](https://github.com/JulianLee310514065/AICUP_audio_2023/#Repository-structure)中有五個`mfcc??_use_all.pth`檔，即為對應`2_AI_CUP_mfcc??.ipynb`數的模型之最好的參數，下載使用即可。

# Training
訓練模型程式寫在`2_AI_CUP_mfcc??.ipynb`中的`Training`部分，`??`代表`n_mfcc`數，這裡值得注意的是，因為這次比賽的各類數量差異太大，故在定義`CrossEntropyLoss`時，我們使用了一個`weight`張量來設定每個類別的權重，以平衡各個類別在訓練過程中的影響。
```python
# Calculate the count of each class.
numberlist = training_df['Disease category'].value_counts().sort_index().to_list()

# model 
model = Network().to(device)

# optimizer
weight = torch.tensor([1/numberlist[0], 1/numberlist[1], 1/numberlist[2], 1/numberlist[3], 1/numberlist[4]]).to(device)
criterion = nn.CrossEntropyLoss(weight=weight)
optimizer = SGD(model.parameters(), lr=0.01, weight_decay= 0.0001)
```


# Inference (public、private data)
```
介紹
```

# Ensemble
```
```

# Reproducing submission
若要重現最終提交結果，可以做以下步驟:
1. 完整跑 [Prepared dataset](https://github.com/JulianLee310514065/AICUP_audio_2023/#Prepared-dataset)
2. 依序跑五個`2_AI_CUP_mfccxx.ipynb`，但不須跑`Training`部分
3. 完整跑 [Ensemble](https://github.com/JulianLee310514065/AICUP_audio_2023/#Ensemble)


若需最後高分之上傳結果，也在[repository](https://github.com/JulianLee310514065/AICUP_audio_2023/#Repository-structure)中，為`submission.csv`

# Acknowledgement
前處理:
* [AI CUP 2023 春季賽【多模態病理嗓音分類競賽】巡迴課程](https://www.youtube.com/playlist?list=PLk_m5EiRQRF3j35iw-93Wh4cGa5fy41gu)
* [Deep Learning Audio Classification](https://medium.com/analytics-vidhya/deep-learning-audio-classification-fcbed546a2dd)

模型架構:
* [crnn-audio-classification](https://github.com/ksanjeevan/crnn-audio-classification#models)
* [audio_classification](https://github.com/harmanpreet93/audio_classification)

# Citation
```
@misc{
    title  = {AICUP_audio_2023},
    author = {Chang-Yi Lee}
    url    = {https://github.com/JulianLee310514065/AICUP_audio_2023},
    year   = {2023}
}
```
