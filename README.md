# AICUP_audio_2023
> **完整程式說明報告**[**(Link)**](https://github.com/JulianLee310514065/AICUP_audio_2023/blob/main/AI_CUP_2023%E5%A4%9A%E6%A8%A1%E6%85%8B%E7%97%85%E7%90%86%E5%97%93%E9%9F%B3%E5%88%86%E9%A1%9E_%E5%A0%B1%E5%91%8A_TEAM_3134.pdf)
>
> **多模態病理嗓音分類競賽**[**(Link)**](https://tbrain.trendmicro.com.tw/Competitions/Details/27)

此repository為 AICUP-2023 多模態病理嗓音分類競賽，用以上傳程式以及提交的答案用的。其比賽內容為主辦方提供五種類型的人的發聲音檔，分別是嗓音誤用、聲帶麻痺、聲帶腫瘤、聲帶閉合不全與正常組，並還提供受測者之生理資料供我們利用，參賽者需要透過訓練模型，儘可能準確地預測每一組未知資料，努力提高Unweighted Average Recall（UAR）指標。

## 比賽重點與遇到之問題
* 各族群資料筆數「極」為不平均，最多的跟最少的族群資料量相差至17倍
* 官方提供了各受測者的生理資料，需要斟酌聲音與生理資料的權重比例

## 詳細內容

* 為確保資料等長，僅選取音頻的第一秒，並利用MFCC對聲音訊號進行前處理
* 建立雙輸入之深度學習模型，同時輸入聲音資訊與生理資訊，輸出特徵數量比例為1:2，且卷積神經網路之卷積核的大小為3*10，這是一種非典型的長方形卷積核
* 建立損失函數時，加入權重以解決族群間數量差距過大問題
* 使用CELU作為模型的激活函數，而非RELU。
* 選擇隨機梯度下降(SGD)作為優化器
* 透過調整參數，訓練出五個有差異性的模型，並做Voting Ensemble來提高預測準確度

# Leaderboard
|Public Score|Public Rank|Private Score|Private Rank|
|-|-|-|-|
|0.600654|7 / 371|0.607568|6 / 371|


 


---
## 更新
最後結果:

**第四名** + **趨勢科技潛力獎**

<div align="center">
	<img src="https://github.com/JulianLee310514065/AICUP_audio_2023/blob/main/rank.png" alt="final" width="600">
</div>

<div align="center">
	<img src="https://github.com/JulianLee310514065/AICUP_audio_2023/blob/main/sci_.png" alt="final" width="600">
</div>

# Getting the code
```
git clone https://github.com/JulianLee310514065/AICUP_audio_2023.git
```

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
模型訓練程式寫在`2_AI_CUP_mfcc??.ipynb`中的`Training`部分，`??`代表`n_mfcc`數，這裡值得注意的是，因為這次比賽的各類數量差異太大，故在定義`CrossEntropyLoss`時，我們使用了一個`weight`張量來設定每個類別的權重，以平衡各個類別在訓練過程中的影響。
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
在驗證模型的部分，我們首先從讀取最佳模型開始，然後分別運行`Public data`、`Private data`，最後將它們合併在一起，以`2_AI_CUP_mfcc13.ipynb`做舉例。
```python
# Load model
model.load_state_dict(torch.load("{}.pth".format("mfcc13_use_all")))

# Predict public data
data_df = pd.read_csv(r'..\Public_Testing_Dataset\test_datalist_public.csv')
...
y_pub = [x.numpy() for x in pub_save]
y_pub[:5]

# Predict private data
data_private_df = pd.read_csv(r'..\Private_Testing_Dataset\test_datalist_private.csv')
...
y_pri = [x.numpy() for x in pub_save_private]
y_pri[:5]

# Combine and save
y_all = y_pub + y_pri
mmffcc13 = np.array(y_all)
np.save('output_mfcc13.npy', mmffcc13)
```

# Ensemble
結果集成的方面我使用的是機率相加，及五種模型與測出來的機率相加，然後取最高的類別作為該聲音資訊的預測結果，程式於`3_Ensemble_pun_pri.ipynb`中。
```python
# Load numpy file
out1 = np.load('output_mfcc13.npy')
out2 = np.load('output_mfcc17.npy')
out3 = np.load('output_mfcc21.npy')
out4 = np.load('output_mfcc30.npy')
out5 = np.load('output_mfcc50.npy')

# Ensemble
out_all = out1 + out2 + out3 + out4 + out5
predict_out = out_all.argmax(1)
```

# Reproducing submission
若要重現最終提交結果，可以做以下步驟:
1. 完整跑 [Prepared dataset](https://github.com/JulianLee310514065/AICUP_audio_2023/#Prepared-dataset)
2. 依序跑五個`2_AI_CUP_mfccxx.ipynb`，但不須跑`Training`部分
3. 完整跑 [Ensemble](https://github.com/JulianLee310514065/AICUP_audio_2023/#Ensemble)，即可得到最後結果。


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
