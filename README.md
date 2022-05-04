# Multi modal EmotionClassifier

## Introduction
인간과 인공지능 사이의 더 자연스러운 교감을 위해, 인간의 감정을 이해하는 것이 우선이 되어야 한다. 이러한 기술의 개발을 위해 [2022 휴먼이해 인공지능 논문경진대회]에서 제공한 KEMDy19 멀티모달 데이터셋을 활용하여 인간의 감정을 인식하는 모델을 개발 및 훈련을 하였다. 활용한 데이터셋은 한국어 멀티모달 감정 데이터셋으로, 발화음성, 발화의 문맥적 의미, 바이오 데이터(EDA, ECG, Temp)와 발화자의 감정과의 연관성 분석을 위해 수집한 멀티모달 데이터셋이다. (Emotion Recognition in Multi-Domain Datasets. Sensors 2021, 21, 1579. https://doi.org/10.3390/s21051579)  
우리는 더 정확하고 세밀한 감정인식 모델 개발을 연구하기 위해 데이터셋의 3개의 모달리티를 모두 활용하였며, 바이오 데이터에서도 EDA, ECG, Temp 데이터를 모두 사용하였다. 또한 더 나은 개발환경 위해 google drive와 google colab pro를 활용하여 작업을 진행하였다.

## Environment
* windows
* google colab

## library
모든 작업은 구글 코랩 환경에서 실행되었음    
추가적으로 설치한 라이브러리인 Mecab의 설치 코드는 Text_emotion_detector.ipynb 파일에 있음

## 파일 생성 코드가 실행이 되지않을 경우 필요한 파일들을 다운로드 받아 사용하기
### 전처리 시 생성된 데이터(text & bio & wav)별 파일들 (4.16GB) (약 11분 소요)
[데이터별 전처리파일 다운로드 링크](https://drive.google.com/file/d/1wdRKR-NTh-rGIYa-bn6uxwtJFZwQvb4F/view?usp=sharing)  
폴더 구성  
**데이터별_전처리파일**  
 |__Bio 전처리 파일들   
 |       |__Bio_data.zip (Segment ID만 존재하는 EDA & ECG & Temp 데이터셋)  
 |       |__df_all_bio.csv  
 |       |__df_all_EDA.csv  
 |       |__df_all_label.csv  
 |       |__merged_bio_text.csv  
 |  
 |__Text 전처리 파일들  
 |       |__df_all_label.csv  
 |       |__df_all_txt.csv  
 |       |__df_label.csv  
 |       |__merged_seg_text.txt  
 |       
 |  
 |__Wav 전처리 파일들   
           |__merged_wav_folder.zip (.wav 파일만 모아둔 wav 데이터셋)  
           |__all_mfcc.npy  
           |__all_wavSeg.txt  
           |__allwav.csv  
           |__df_all_label.csv  
           |__needed_mfcc.npy (x_data)  
           |__needed_mfcc_y.pkl (y_data)  
 
 
### 학습 시 필요한 파일들 및 pretrained_h5파일 (1.28GB) (약 3분 소요)
[학습시 필요파일 및 h5파일 다운로드 링크](https://drive.google.com/file/d/1jVUqTa2g6xTTnYGznGQA_oBKGCIvR1G9/view?usp=sharing)  
폴더 구성  
**학습시_필요파일_및_h5파일**  
 |__h5_models  
 |       |__best_model_bio.h5  
 |       |__best_model_txt.h5   
 |       |__best_model_wav.h5  
 |  
 |__데이터 별 학습시 필요한 파일  
           |__bio  
           |     |__merged_bio_text.csv  
           |  
           |__text  
           |     |__merged_bio_text.csv  
           |  
           |__wav  
                 |__needed_mfcc.npy (x_data)  
                 |__needed_mfcc_y.pkl (y_data)  
  
# Data Preprocessing (KEMDy19)
### Raw dataset에 대한 전처리
* SegmentID & Label: pandas 라이브러리를 통해 annotation에 있는 label값들과 SegmentID값들을 가져와 df_all_label.csv파일을 생성  
* Label에서 ;(세미콜론)이 있는 label의 경우 앞의 emotion을 추출
* Text: os, glob 함수를 이용하여 wav폴더 내 .txt파일 내 text와 해당 파일명을 함께 merged_seg_text.txt파일을 생성     
* WAV: os, glob, shutil 함수를 이용하여 wav폴더 내 .wav파일을 merged_wav_folder폴더로 copy & paste    
* BIO: EDA와 ECG, Temp 세 가지의 감정 데이터의 Segment ID가 있는 행을 추출하여 평균값을 낸 후 df_all_bio.csv파일로 병합  
<hr/>  

### Text
* `dataframe['columns'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")`과 `dataframe['columns'].replace('', np.nan, inplace=True)`을 통해 한글과 공백을 제외하고 모두 제거함
* `dataframe['columns'].nunique()`과 `merged_data.drop_duplicates(subset=['Seg'], inplace=True)`를 통해 해당 열의 중복되는 샘플의 수를 카운트 하고 제거함 
* `dataframe.isnull()`을 통해 null값을 확인
* `dataframe['tokenized'] = dataframe['text'].apply(lambda x : mecab.morphs(str(x)))` 형태소 분석기인 Mecab을 사용하여 토큰화 작업을 수행
* `merged_data['tokenized'] = merged_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])` 을 통해 불용어와 필요없는 토큰을 제거함
* `tokenizer.fit_on_texts(X_train)`을 통해 데이터 셋에 정수 인코딩을 하여 텍스트를 숫자로 처리함
* `max(len(review) for review in X_train`를 통해 텍스트의 최대 길이를 파악한 후 다른 텍스트들과 길이를 맞추기 위해 padding 작업을 진행함
* Imbalance한 데이터를 해결하기 위해 SMOTE(Synthetic Minority Over-sampling Technique)의 `fit_resample()`메소드를 활용
* `train_test_split()`을 통해 train과 test데이터를 8:2의 비율로 분리
<hr/>  

### Wav
* for 반복문을 통해 음성 데이터 폴더 내 wav파일을 librosa 라이브러리로 로드함 `audio, sr = librosa.load(files, sr=16000)`
* `librosa.feature.mfcc()`을 통해 음성 데이터의 특징을 추출함
  * sr: sampling rate로 defalut=22050Hz
  * n_mfcc: return될 mfcc의 개수를 정해주는 파라미터, default=20
  * n_fft: frame의 length를 결정하는 파라미터
  * hop_length: defalut=10ms, hop_length의 길이만큼 옆으로 가면서 데이터를 읽음
* `sklearn.preprocessing.scale()`을 통해 mfcc를 scaling 처리함
* 음성 데이터의 길이가 다양하므로 padding처리를 통해 길이 조정
* `np.save()`를 통해 전처리가 된 wav mfcc값들을 all_mfcc.npy 파일로 저장하고 다른 데이터와 Segment ID값을 맞춘 후 needed_mfcc.npy파일로 저장
* wav의 정답 데이터를 needed_mfcc_y.pkl로 저장
* Imbalance한 데이터를 해결하기 위해 SMOTE(Synthetic Minority Over-sampling Technique)의 `fit_resample()`메소드를 활용
* `train_test_split()`을 통해 train과 test데이터를 8:2의 비율로 분리
<hr/>  

### Bio
* EDA & ECG & Temp raw 데이터셋에서 Segment ID만 존재하는 행만 추출함
* 각각의 데이터의 평균값을 구하기 위해 `dataframe.groupby('group_column')['value'].agg()`를 통해 평균값 추출
* 각각의 데이터의 정규화 작업을 위해 `scaler.fit_transform(dataframe)`을 통해 정규화 처리
* Imbalance한 데이터를 해결하기 위해 SMOTE(Synthetic Minority Over-sampling Technique)의 `fit_resample()`메소드를 활용
* `train_test_split()`을 통해 train과 test데이터를 8:2의 비율로 분리

# Getting Started (colab에서 실행)
### Preprocessing_data.ipynb 또는 Preprocessing_data.py 파일을 통해 전처리 작업 수행  
  * text 작업을 위해 Mecab 라이브러리를 설치해야 하는데, 코랩이 아닌 로컬에서 할 경우는 수동 설치를 해야 하는 번거로움이 있어 colab에서 작업하는 것을 권장함(Preprocessing_data.py 내 Mecab 설치 가이드 링크 존재)
  * 코랩 내에서 merged_seg_text.txt과 all_wavSeg.txt을 생성할 경우, text파일에 순서가 랜덤하게 담기는 이슈가 발생하기 때문에 해당 파일을 생성하려는 경우 해당 부분만은 로컬에서 실행시키기를 권장함
  * preprocessing_data를 통해 생성된 모든 파일은 readme 상단의 *데이터별 전처리파일 다운로드 링크*를 통해 다운로드 받을 수 있음
<hr/>  

### google colab을 통한 각각의 모델 train
1. **각각의 모델 학습 및 .h5 저장**
* Text_emotion_detector.ipynb을 통해 text모델을 학습 및 테스트할 수 있다.
  * Text 모델을 학습시킬 때 필요한 merged_bio_text.csv파일의 경로를 설정해 준 뒤 각 터미널을 실행시킨다.

* Bio_emotion_detector.ipynb을 통해 bio모델을 학습 및 테스트할 수 있다.
  * Bio 모델을 학습시킬 때 필요한 merged_bio_text.csv파일의 경로를 설정해 준 뒤 각 터미널을 실행시킨다.

* Wav_emotion_detector.ipynb을 통해 wav모델을 학습 및 테스트할 수 있다.
  * Wav 모델을 학습시킬 때 필요한 needed_mfcc.npy(X_data)와 needed_mfcc_y.pkl(Y_data)파일의 경로를 설정해 준 뒤 각 터미널을 실행시킨다.
* 학습에 필요한 파일 및 사전학습된 모델은 readme 상단의 *학습시 필요파일 및 h5파일 다운로드 링크*를 통해 다운로드 받을 수 있음
2. **각각의 모델의 예측값을 Ensemble 모델에 활용하여  최종 성능 평가**

* Ensemble_emotion.ipynb을 통해 ensemble 모델을 테스트할 수 있다.
  * 각각의 모델(Text & Bio & Wav)의 h5파일 경로를 설정해 준 뒤 각 터미널을 실행시킨다.
  * 세 개의 모델의 predict값을 출력해준 뒤, ensemble 모델의 Soft voting을 거쳐 성능을 향상시킬 수 있다.

