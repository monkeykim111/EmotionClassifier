# Multi modal EmotionClassifier

## Abstract
프로젝트의 진행 및 플로우에 대한 간략한 요약 및 설명

1. 코드 설명  
사용하신 라이브러리 및/또는 데이터 분석 결과에 대한 내용 (별도로 필요한 라이브러리가 있을 경우 소스코드 내에 해당 라이브러리 설치하는 코드 기재)

## Environment
* windows 운영체제
* google colab

## library
모든 작업은 구글 코랩 환경에서 실행되었음 
추가적으로 설치한 라이브러리인 Mecab의 설치 코드는 Text_emotion_detector.ipynb 파일에 있음

## 파일 생성 코드가 실행이 되지않을 경우 필요한 파일들 설치하기
### 전처리 시 생성된 데이터 별(text & bio & wav) 파일들 (4.16GB)
https://drive.google.com/file/d/1wdRKR-NTh-rGIYa-bn6uxwtJFZwQvb4F/view?usp=sharing

### 학습 시 필요한 파일들 및 h5파일 (1.28GB)
https://drive.google.com/file/d/1jVUqTa2g6xTTnYGznGQA_oBKGCIvR1G9/view?usp=sharing

# Data Preprocessing
### raw dataset에 대한 전처리
* SegmentID & Label: pandas 라이브러리를 통해 annotation에 있는 label값들과 SegmentID값들을 가져와 df_all_label.csv파일을 생성  
* Label에서 ;(세미콜론)이 있는 label의 경우 앞의 emotion을 추출하는 함수 정의
* Text: os, glob 함수를 이용하여 wav폴더 내 .txt파일 내 text와 해당 파일명을 함께 merged_seg_text.txt파일을 생성
* WAV: os, glob, shutil 함수를 이용하여 wav폴더 내 .wav파일을 merged_wav_folder폴더로 copy & paste  
* BIO: EDA와 ECG, Temp 세 가지의 감정 데이터의 Segment ID가 있는 행을 추출하여 평균값을 낸 후 df_all_bio.csv파일로 병합

### Text emotion detector
#### Colab에 Mecab 설치
```
!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
%cd Mecab-ko-for-Google-Colab
!bash install_mecab-ko_on_colab190912.sh
```

* `dataframe['columns'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")`과 `dataframe['columns'].replace('', np.nan, inplace=True)`을 통해 한글과 공백을 제외하고 모두 제거함
* `dataframe['columns'].nunique()`과 `merged_data.drop_duplicates(subset=['Seg'], inplace=True)`를 통해 해당 열의 중복되는 샘플의 수를 카운트 하고 제거함 
* `dataframe.isnull()`을 통해 null값을 확인
* `dataframe['tokenized'] = dataframe['text'].apply(lambda x : mecab.morphs(str(x)))` 형태소 분석기인 Mecab을 사용하여 토큰화 작업을 수행
* `merged_data['tokenized'] = merged_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])` 을 통해 불용어와 필요없는 토큰을 제거함
* `tokenizer.fit_on_texts(X_train)`을 통해 데이터 셋에 정수 인코딩을 하여 텍스트를 숫자로 처리함
*  `max(len(review) for review in X_train`를 통해 텍스트의 최대 길이를 파악한 후 다른 텍스트들과 길이를 맞추기 위해 padding 작업을 진행함
* 다중분류를 하기 위해 Label 데이터를 one hot encoding 처리를 하여 category 형으로 생성함
* `smote.fit_resample()`을 통해 imbalance한 데이터를 증강
* `train_test_split()`을 통해 train과 test데이터를 8:2의 비율로 분리

### Wav
* for 반복문을 통해 음성 데이터 폴더 내 wav파일을 librosa 라이브러리로 로드함 `audio, sr = librosa.load(files, sr=16000)`
* `librosa.feature.mfcc()`을 통해 음성 데이터의 특징을 추출함
  * sr: sampling rate로 defalut=22050Hz
  * n_mfcc: return될 mfcc의 개수를 정해주는 파라미터, default=20
  * n_fft: frame의 length를 결정하는 파라미터
  * hop_length: defalut=10ms, hop_length의 길이만큼 옆으로 가면서 데이터를 읽음
* `sklearn.preprocessing.scale()`을 통해 mfcc를 scaling 처리함
* 음성 데이터의 길이가 다양하므로 padding처리를 통해 길이 조정
* 다중분류를 하기 위해 Label 데이터를 one hot encoding 처리를 하여 category 형으로 생성함
* `np.save()`를 통해 전처리가 된 wav mfcc값들을 all_mfcc.npy 파일로 저장하고 다른 데이터와 Segment ID값을 맞춘 후 needed_mfcc.npy파일로 저장
* wav의 정답 데이터를 needed_mfcc_y.pkl로 저장
* `smote.fit_resample()`을 통해 imbalance한 데이터를 증강
* `train_test_split()`을 통해 train과 test데이터를 8:2의 비율로 분리

### Bio
* 각각의 데이터의 평균값을 구하기 위해 `dataframe.groupby('group_column')['value'].agg()`를 통해 평균값 추출
* 각각의 데이터의 정규화 작업을 위해 `scaler.fit_transform(dataframe)`을 통해 정규화 처리
* 다중분류를 하기 위해 Label 데이터를 one hot encoding 처리를 하여 category 형으로 생성함
* `smote.fit_resample()`을 통해 imbalance한 데이터를 증강
* `train_test_split()`을 통해 train과 test데이터를 8:2의 비율로 분리

# Getting Started
### Preprocessing_data.ipynb 또는 Preprocessing_data.py 파일을 통해 전처리 작업 수행  
  * text 작업을 위해 Mecab 라이브러리를 설치해야 하는데, 코랩이 아닌 로컬에서 할 경우는 수동 설치를 해야 하는 번거로움이 있어 colab에서 작업하는 것을 권장함(Preprocessing_data.py 내 링크 존재)
  * 코랩 내에서 merged_seg_text.txt과 all_wavSeg.txt을 생성할 경우, text파일에 순서가 랜덤하게 담기는 이슈가 발생하기 때문에 해당 파일을 생성하려는 경우 해당 부분만은 로컬에서 실행시키기를 권장함
  * preprocessing_data를 통해 생성된 모든 파일은 설치링크를 통해 다운로드 받을 수 있음

### google colab을 통한 각각의 모델 train
1. 각각의 모델 학습 및 .h5 저장
* Text_emotion_detector.ipynb을 통해 text모델을 학습 및 테스트할 수 있다.
  * Text 모델을 학습시킬 때 필요한 merged_bio_text.csv파일의 경로를 설정해 준 뒤 각 터미널을 실행시킨다.

* Bio_emotion_detector.ipynb을 통해 bio모델을 학습 및 테스트할 수 있다.
  * Bio 모델을 학습시킬 때 필요한 merged_bio_text.csv파일의 경로를 설정해 준 뒤 각 터미널을 실행시킨다.

* Wav_emotion_detector.ipynb을 통해 wav모델을 학습 및 테스트할 수 있다.
  * Wav 모델을 학습시킬 때 필요한 needed_mfcc.npy(X_data)와 needed_mfcc_y.pkl(Y_data)파일의 경로를 설정해 준 뒤 각 터미널을 실행시킨다.

2. 각각의 모델의 예측값을 Ensemble 모델에 활용하여  최종 성능 평가

* Ensemble_emotion.ipynb을 통해 ensemble 모델을 테스트할 수 있다.
  * 각각의 모델(Text & Bio & Wav)의 predict값을 출력해준 뒤, ensemble 모델의 Soft voting을 거쳐 성능을 향상시킬 수 있다.

