# Mecab library 설치하기
# Colab에 Mecab 설치
'''
!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
%cd Mecab-ko-for-Google-Colab
!bash install_mecab-ko_on_colab190912.sh
'''

# window에 Mecab 설치 (아래 링크 참조(택1))
'''
# https://velog.io/@jyong0719/konlpy-mecab-%EC%84%A4%EC%B9%98-window
# https://uwgdqo.tistory.com/363
'''

# 라이브러리 import
import os
import glob
import pandas as pd
import shutil
import warnings

# text preprocessing
from konlpy.tag import Mecab

# Bio preprocessing
from sklearn.preprocessing import MinMaxScaler

# Wav preprocessing
import librosa
import numpy as np
import pickle
import sklearn
from sklearn.preprocessing import scale

############ Label preprocessing
# Annotation 폴더 내 cvs에서 label 데이터 추출
annotation = 'Your_dataset_path/Human_understand/KEMDy19/annotation/*.csv'
folders = glob.glob(annotation)
df_all_label = pd.DataFrame()

for files in folders:
    Label = pd.read_csv(files, usecols=[9, 10])
    df_all_label = pd.concat([df_all_label, Label])

df_all_label.rename(columns={'Segment ID':'Seg', 'Total Evaluation':'Label'}, inplace=True)
df_all_label = df_all_label.drop([0])
print('df_all_label', df_all_label)
print('==============Label 데이터 추출완료==============')

# Lable csv 파일로 저장
df_all_label.to_csv("df_label.csv", mode='w')
print('==============Label 데이터 저장완료==============')

# 저장한 Label csv파일 read
label= pd.read_csv('Your_dataset_path/Human_understand/KEMDy19/df_label.csv')

# 감정 레이블에 ;(세미콜론)있는 경우 앞의 감정을 추출하는 함수 정의
def delSemi(x):
  if ";" in x:
    idx_number = x.find(";")
    return x[:idx_number]
  else:
    return x

# label 컬럼에 apply함수를 적용
label['Label'] = label['Label'].apply(lambda x:delSemi(x))
print('label', label)

# Lable csv 파일로 저장 (세미콜론 앞의 감정이 추출된 Label 파일)
label.to_csv("df_all_label.csv", mode='w')
print('==============Label데이터 cav파일로 저장완료==============')

############ TEXT preprocessing
# 폴더 내에 있는 .txt 파일만 읽고 하나의 .txt파일에 입력하기
targetPattern = 'Your_dataset_path/Human_understand/KEMDy19/wav/**/**/*.txt'
allTextFile = glob.glob(targetPattern)

# merged_seg_text.txt파일에 raw text 입력
mergedText = open('merged_seg_text.txt', 'w', encoding="UTF-8")
for i in range(len(allTextFile)):
    myText = allTextFile[i]
    first = myText.rfind('Sess')
    last = myText.find('.txt')
    sessID = myText[first:last]
    myText = open(myText, 'r', encoding="UTF-8")
    text = sessID + ',' + myText.readline()
    mergedText.write(text)
mergedText.close()
print('==============Text 데이터 추출완료==============')

#text 데이터와 label 데이터와 병합 후 각 row에 중복되는 값 제거
df_all_text = pd.read_csv('Your_dataset_path/Human_understand/KEMDy19/merged_seg_text.txt', names=['Seg', 'text'])
label = pd.read_csv('Your_dataset_path/Human_understand/KEMDy19/df_all_label.csv', names=['Seg', 'Label'])

# text와 label 데이터를 Segment ID를 기준으로 병합하기
df_all_txt_label = pd.merge(label, df_all_text, on='Seg')

# 정규 표현식을 사용하여 한글을 제외한 단어 제거 및 공백 제거하기
df_all_txt_label['text'] = df_all_txt_label['text'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
df_all_txt_label['text'].replace('', np.nan, inplace=True)
df_all_txt_label = df_all_txt_label.dropna(how='any')

# 각 컬럼에서 중복되는 row 삭제
df_all_txt_label['Seg'].nunique(), df_all_txt_label['Label'].nunique(), df_all_txt_label['text'].nunique()
df_all_txt_label.drop_duplicates(subset=['Seg'], inplace=True)
print('df_all_txt_label', df_all_txt_label)

# null 값이 있는지 확인하기
print('null값이 있나요?', df_all_txt_label.isnull().values.any())

#mecab을 이용하여 text 컬럼에 tokenize 적용
# Mecab init
mecab = Mecab()
# 불용어 정의
stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게']

# text에 tokenize 적용
df_all_txt_label['tokenized'] = df_all_txt_label['text'].apply(lambda x : mecab.morphs(str(x)))
df_all_txt_label['tokenized'] = df_all_txt_label['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])

# df_all_txt csv 파일로 저장
df_all_txt_label.to_csv('df_all_txt.csv', mode='w', encoding="utf-8-sig")
print('==============Text 데이터 cav파일로 저장완료==============')

############ Bio preprocessing
# Segment ID가 없는 row제거 후 EDA, ECG, Temp.csv파일 load

# EDA 데이터 load 후 각 segment ID의 평균값 적용
folders=r'Your_dataset_path/Human_understand/KEMDy19/EDA/**/*/*.csv'
folders=glob.glob(folders)
df_all_EDA=pd.DataFrame()

for files in folders:
    EDA=pd.read_csv(files,encoding='cp949', names=["EDA", "b", "c", "Seg"] ,usecols=['EDA','Seg'])
    mean_EDA=EDA.groupby('Seg')['EDA'].agg(**{'mean_EDA':'mean'}).reset_index()
    df_all_EDA=pd.concat([df_all_EDA, mean_EDA])
print('df_all_EDA', df_all_EDA)
print('==============EDA 데이터 추출완료==============')

# EDA csv 파일로 저장
df_all_EDA.to_csv("df_all_EDA.csv", mode='w')
print('==============EDA 데이터 cav파일로 저장완료==============')

# EDA 파일과 Label 파일 병합하기
EDA=pd.read_csv('Your_dataset_path/Human_understand/KEMDy19/df_all_EDA.csv',encoding='cp949')
label= pd.read_csv('Your_dataset_path/Human_understand/KEMDy19/df_all_label.csv', names=['Seg','Label'])

# 감정 레이블에 ;(세미콜론)있는 경우 앞의 감정을 추출
label['Label'] = label['Label'].apply(lambda x:delSemi(x))

# EDA data와 label data 병합하기
merged_EDA_label = pd.merge(EDA, label, on='Seg')

# ECG 데이터 load 후 각 segment ID의 평균값 적용
folders= 'Your_dataset_path/Human_understand/KEMDy19/ECG/**/*/*.csv'
folders=glob.glob(folders)
df_all_ECG=pd.DataFrame()

for files in folders:
    ECG=pd.read_csv(files,encoding='cp949', names=["ECG", "b", "c", "Seg"] ,usecols=['ECG','Seg'])
    mean_ECG=ECG.groupby('Seg')['ECG'].agg(**{'mean_ECG':'mean'}).reset_index()
    df_all_ECG=pd.concat([df_all_ECG, mean_ECG])
print('df_all_ECG', df_all_ECG)
print('==============ECG 데이터 추출완료==============')

# Temp 데이터 load후 각 segment ID의 평균값 적용
folders=r'Your_dataset_path/Human_understand/KEMDy19/TEMP/**/*/*.csv'
folders=glob.glob(folders)
df_all_Temp=pd.DataFrame()

for files in folders:
    Temp=pd.read_csv(files,encoding='cp949', names=["Temp", "b", "c", "Seg"] ,usecols=['Temp','Seg'])
    mean_Temp=Temp.groupby('Seg')['Temp'].agg(**{'mean_Temp':'mean'}).reset_index()
    df_all_Temp=pd.concat([df_all_Temp, mean_Temp])
print('df_all_Temp', df_all_Temp)
print('==============Temp 데이터 추출완료==============')

df_all_bio=pd.DataFrame()
df_all_bio1=pd.DataFrame()
df_all_bio2=pd.DataFrame()

# EDA & ECG & Temp data 병합
df_all_bio1 = pd.merge(merged_EDA_label, df_all_ECG, on='Seg')
df_all_bio2 = pd.merge(df_all_Temp, df_all_bio1, on='Seg')

# df_all_bio.csv 파일 저장
df_all_bio2.to_csv("df_all_bio.csv", mode='w')
print('==============BIO 데이터 cav파일로 저장완료==============')

# csv로 저장된 df_all_bio csv 파일 read
df_all_bio=pd.read_csv('Your_dataset_path/Human_understand/KEMDy19/df_all_bio.csv', usecols=['Seg', 'mean_EDA', 'mean_ECG', 'mean_Temp', 'Label'])

# 감정 레이블에 ;(세미콜론)있는 경우 앞의 감정을 추출
df_all_bio['Label'] = df_all_bio['Label'].apply(lambda x:delSemi(x))

# 0 ~ 1 사이의 값으로 정규화 세팅
scaler=MinMaxScaler(feature_range=(0,1))

# 정규화를 적용하기 위해 각각의 Bio csv 파일 read 한 후 정규화 적용하기
scaled_mean_EDA=pd.read_csv("Your_dataset_path/Human_understand/KEMDy19/df_all_bio.csv",usecols=["mean_EDA"])
scaled_mean_ECG=pd.read_csv("Your_dataset_path/Human_understand/KEMDy19/df_all_bio.csv",usecols=["mean_ECG"])
scaled_mean_Temp=pd.read_csv("Your_dataset_path/Human_understand/KEMDy19/df_all_bio.csv",usecols=["mean_Temp"])

df_all_bio["mean_EDA"]=scaler.fit_transform(scaled_mean_EDA)
df_all_bio["mean_ECG"]=scaler.fit_transform(scaled_mean_ECG)
df_all_bio["mean_Temp"]=scaler.fit_transform(scaled_mean_Temp)

# 최소값을 적용하여 중복값 제거
df_all_bio = df_all_bio.groupby('Seg').min()

# scaled_df_all_bio.csv로 저장
df_all_bio.to_csv("scaled_df_all_bio.csv", mode="w")
print('==============정규화 처리한 BIO 데이터 cav파일로 저장완료==============')

# scaled_df_all_bio.csv와 df_all_tex.csv를 read
scaled_df_all_bio=pd.read_csv('Your_dataset_path/Human_understand/KEMDy19/scaled_df_all_bio.csv')
df_all_txt = pd.read_csv('Your_dataset_path/Human_understand/KEMDy19/df_all_txt.csv', usecols=['Seg', 'tokenized'])

# bio data와 text data를 병합
merged_bio_text = pd.merge(scaled_df_all_bio, df_all_txt, on='Seg', how='left')

# bio와 text가 병합된 data의 label 컬럼에 one-hot-encoding 적용 후 csv파일로 저장하기
merged_bio_text = pd.get_dummies(merged_bio_text, columns=['Label'])
merged_bio_text.to_csv("merged_bio_text.csv", mode="w", encoding='utf-8-sig')
print('==============BIO 데이터와 Text 병합한 데이터 cav파일로 저장완료==============')

###### Wav preprocessing
# Wav 파일내 .wav파일들을 하나의 폴더에 copy하기
targetPattern = 'Your_dataset_path/Human_understand/KEMDy19/wav/**/**/*.wav'
allWavFile = glob.glob(targetPattern)

for wav_file in allWavFile:
    shutil.copy(wav_file, 'Your_dataset_path/Human_understand/merged_wav_folder')
print('==============Label 데이터 추출완료==============')

# 하나로 모인 .wav 파일 로드 및 MFCC 추출 후 .npy 파일로 저장
folders= 'Your_dataset_path/Human_understand/merged_wav_folder/*.wav'
folders=glob.glob(folders)

# mfcc 추출시 발생되는 warning 제거하기
warnings.filterwarnings(action='ignore')

append_list=[]
extend_list=[]
for files in folders:
    audio, sr = librosa.load(files, sr=16000)
    #mfcc추출 파라미터 설정
    mfcc = librosa.feature.mfcc(audio, sr=16000, n_mfcc=100, n_fft=400, hop_length=160) 
    #전처리 scaling
    mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
    #데이터의 길이를 5.35초로 자르고 5.35초보다 작을 경우에만 패딩 작업
    pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))
    #0.46초=40,5.35초=(about)465
    padded_mfcc = pad2d(mfcc, 465)
    
    append_list.append(padded_mfcc)
    extend_list.extend(append_list)
    append_list.clear()

extend_list_array=np.array(extend_list)

# mfcc가 담긴 extend_list_array를 .npy로 저장
np.save(r'Your_dataset_path/Human_understand/KEMDy19/all_mfcc.npy', extend_list_array)
print('==============MFCC 데이터 npy파일로 저장완료==============')

# wav파일의 segment ID값을 저장하기 위해 all_wavSeg.txt 파일 생성
'''
1. 생성한 mfcc(all_mfcc.npy)파일에는 segment ID가 없으므로 추후 감정label 추출을 위해 segment ID가 필요함
2. wav 파일을 기준으로 mfcc를 생성했기 때문에,
해당 파일의 이름이 segment ID라는 점을 고려하여 wav file 이름을 추출함
'''

targetPattern = 'Your_dataset_path/Human_understand/KEMDy19/wav/**/**/*.wav'
allTextFile = glob.glob(targetPattern)

# wav 파일구성 순서대로 segment ID를 추출하여 하나의 text파일에 입력
mergedText = open('all_wavSeg.txt', 'w', encoding="UTF-8")

for i in range(len(allTextFile)):
    myText = allTextFile[i]
    first = myText.rfind('Sess')
    last = myText.find('.wav')
    sessID = myText[first:last]
    myText = open(myText, 'r', encoding="UTF-8")
    text = sessID + ',' + '\n'
    mergedText.write(text)
mergedText.close()
print('==============WAV Segment ID 데이터 추출완료==============')

# wav segment ID와 Label data를 read한 후 병합하기
all_wavSeg = pd.read_csv('Your_dataset_path/Human_understand/KEMDy19/all_wavSeg.txt', names=['Seg', 'a'])
label= pd.read_csv('Your_dataset_path/Human_understand/KEMDy19/df_all_label.csv')
allwav = pd.merge(all_wavSeg, label, on='Seg', how='left')

# 병합된 wav segment ID와 Label data를 csv로 저장
allwav.to_csv("allwav.csv", mode="w")
print('==============병합한 WAV Segment ID 데이터와 Lable 데이터 저장완료==============')

# 결측값을 제거하기 위해 bio['seg']를 기준으로 데이터 취합
# merged_bio_text.csv를 read
bio_seg=pd.read_csv('Your_dataset_path/Human_understand/KEMDy19/merged_bio_text.csv', usecols=["Seg"])

# all_wavSeg 파일 read후 list로 변형
wav_seg=pd.read_csv("Your_dataset_path/Human_understand/KEMDy19/all_wavSeg.txt",names=['Seg'])
wav_seg_list=wav_seg["Seg"].to_list()

# bio의 segment ID를 wav_seg_list에서 찾고 해당 index를 mfcc_index 리스트에 저장 
mfcc_index=[]
for i in range(len(bio_seg)):
    #search에 segID 들어감
    search=bio_seg.loc[i,['Seg']].values 
    if search in wav_seg_list:
        mfcc_index.append(wav_seg_list.index(search))
str(mfcc_index)

# mfcc_index 리스트에 저장된 index를 기준으로  
# 앞서 생성한 all_mfcc에서 해당 index의 mfcc를 추출하여 needed mfcc에 저장하기
all_mfcc = np.load('Your_dataset_path/Human_understand/all_mfcc.npy')
all_mfcc.tolist()

needed_mfcc=[]
for i in mfcc_index:
    needed_mfcc.append(all_mfcc[i])
needed_mfcc_array=np.array(needed_mfcc)
np.save(r'Your_dataset_path/Human_understand/needed_mfcc.npy', needed_mfcc_array)
print('==============추출한 인덱스를 통해 필요한 MFCC 데이터를 뽑은 후 npy파일로 저장==============')

# y데이터(정답 데이터) 생성
mfcc_y = pd.read_csv(r'C:/Users/user/Downloads/KEMDy19/allwav.csv', usecols=["Label"])
needed_mfcc_y=[]
for i in mfcc_index:
    needed_mfcc_y.append(mfcc_y.loc[i,['Label']].values)

# pickle file로 저장
with open('needed_mfcc_y.pkl','wb') as f:
    pickle.dump(needed_mfcc_y, f)
print('==============MFCC 데이터와 매칭되는 label 데이터 pickle파일로 저장완료==============')

