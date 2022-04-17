# Multi modal EmotionClassifier

## Abstract
프로젝트의 진행 및 플로우에 대한 간략한 요약 및 설명

1. 코드 설명
사용하신 라이브러리 및/또는 데이터 분석 결과에 대한 내용 (별도로 필요한 라이브러리가 있을 경우 소스코드 내에 해당 라이브러리 설치하는 코드 기재)

## Environment
* windows os
* google colab
* python 3.8

## libraries
필요한 라이브러리는 requirements.txt 참고  
`pip3 install -r requirements.txt`
#### Colab에 Mecab 설치
```
!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
%cd Mecab-ko-for-Google-Colab
!bash install_mecab-ko_on_colab190912.sh
```

## Data Preprocessing
데이터의 정제/통합/정리/변환 등 데이터 전처리 과정 및 결과에 대한 내용
### raw dataset에 대한 전처리

* SegmentID & Label: pandas 라이브러리를 통해 annotation에 있는 label값들과 SegmentID값들을 가져와 Segment csv파일을 생성  
* Label에서 ;(세미콜론)이 있는 label의 경우 앞의 것을 따르도록 처리하였음
* text: os, glob 함수를 이용하여 wav폴더 내 .txt파일의 text와 해당 파일명을 함께 text파일로 저장  
* WAV: os, glob, shutil 함수를 이용하여 wav폴더 내 .wav파일을 새로운 폴더로 copy & paste함  
* BIO: EDA와 ECG, Temp 세 가지의 감정 데이터를 사용하였음
* 
*

### Text
* `dataframe['columns'].nunique()`과 `merged_data.drop_duplicates(subset=['Seg'], inplace=True)`를 통해 해당 열의 중복되는 샘플의 수를 카운트 하고 제거함 
* `dataframe.isnull()`을 통해 null값을 확인
* `train_test_split()`을 통해 train과 test 데이터를 8:2의 비율로 분리함
* `dataframe['columns'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")`과 `dataframe['columns'].replace('', np.nan, inplace=True)`을 통해 한글과 공백을 제외하고 모두 제거함
* `dataframe['tokenized'] = dataframe['text'].apply(lambda x : mecab.morphs(str(x)))` 형태소 분석기인 Mecab을 사용하여 토큰화 작업을 수행
* `merged_data['tokenized'] = merged_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])` 을 통해 불용어와 필요없는 토큰을 제거함
* `tokenizer.fit_on_texts(X_train)`을 통해 데이터 셋에 정수 인코딩을 하여 텍스트를 숫자로 처리함
* `len(tokenizer.word_index)`을 출력하여 등장횟수가 1회인 단어들은 학습에서 배제함
*  ## OOV로 변환한다는데 이게 뭔지 찾아보기
*  `max(len(review) for review in X_train`를 통해 텍스트의 최대 길이를 파악한 후 다른 텍스트들과 길이를 맞추기 위해 padding 작업을 진행함

### Wav
* for 반복문을 통해 음성 데이터 폴더 내 wav파일을 librosa 라이브러리로 로드함 `audio, sr = librosa.load(files, sr=16000)`
* `librosa.feature.mfcc()`을 통해 음성 데이터의 특징을 추출함
  * sr: sampling rate로 defalut=22050Hz
  * n_mfcc: return될 mfcc의 개수를 정해주는 파라미터, default=20
  * n_fft: frame의 length를 결정하는 파라미터
  * hop_length: defalut=10ms, hop_length의 길이만큼 옆으로 가면서 데이터를 읽음
* `sklearn.preprocessing.scale()`을 통해 mfcc를 scaling 처리함
* 음성 데이터의 길이가 다양하므로 padding처리를 통해 길이를 맞춤
* `librosa.display.specshow()`를 통해 시각화하여 padding 처리된 음성 데이터를 시각화함

### Bio
* EDA와 ECG, TEMP 세 가지 데이터에서 Segment ID가 존재하는 데이터만 추출하여 각각의 csv파일을 생성함
* 각각의 Bio 데이터를 dataframe으로 읽고 segment ID를 기준으로 label 데이터와 함께 merge함
* 다중분류를 하기 위해 Label 데이터를 one hot encoding 처리를 하여 category 형으로 생성함
* `scaler.fit_transform(scaled_mean_BIO)`를 적용하여 각각의 EDA, ECG, TEMP 데이터를 0 ~ 1 사이의 값으로 정규화함
* 


## Getting Started
2. 코드 실행 방식에 대한 설명
공유하신 모든 코드는 제 3자가 직접 재현할 수 있도록 최소한의 주석 혹은 가이드 제공

오류 없이 원활한 코드 구동/실행이 이뤄질 수 있도록 가능한 상세한 설명 기입 요망

## Reference



