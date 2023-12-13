import json
import pandas as pd
from os import listdir
from tqdm import tqdm
from konlpy.tag import Komoran, Okt
from gensim.models import word2vec
from gensim.models import KeyedVectors
import numpy as np

Komoran = Komoran()
Okt = Okt()
import matplotlib.pyplot as plt
import seaborn as sns

# 라벨링 데이터 가져오는 함수
def get_data(data, path):
    ids, grades, etypes, essay, levels, scores = [], [], [], [], [], []  # 빈 리스트 생성
    for row in tqdm(data.itertuples(), total=data.shape[0]):  # 진행상황 확인을 위한 tqdm, data frame 값을 빠르게 가져오기 위한 itertuples
        file_str = path + row.file_name  # 파일명

        with open(file_str, 'r', encoding='utf-8') as f:  # json파일 열기
            text = json.load(f)

        txt = text['paragraph'][0]['paragraph_txt']
        paragraph_score = text['score']['paragraph_score'][0]['paragraph_scoreT_avg']  # paragraph score data 가져오기
        essay_score = text['score']['essay_scoreT_avg']  # essay score data 가져오기
        score = paragraph_score + essay_score  # score 더하기

        ids.append(text['info']['essay_id'])  # essay_id
        grades.append(text['rubric']['essay_grade'])  # 학년
        etypes.append(text['info']['essay_type'])  # 글 종류
        essay.append(txt)
        levels.append(text['info']['essay_level'])  # 각 글의 level
        scores.append(score)  # score data

    df = pd.DataFrame({'id': ids, 'grade': grades, 'etype': etypes, 'essay': essay, 'levels': levels,
                       'scores': scores})  # 리스트로 dataframe 만들기
    return df


def load_data(data_type='train'):

    text_type = ['글짓기', '대안제시', '설명글', '주장', '찬성반대']  # 글 5가지 종류
    main_df = pd.DataFrame()  # 빈 데이터프레임

    if data_type == 'test':
        path = './2.Validation/라벨링데이터/'  # 맞는 경로를 지정해줘야 함
    else:
        path = './1.Training/라벨링데이터/'  # 맞는 경로를 지정해줘야 함

    # 파일 리스트로부터 모든 데이터 가져오기
    for i in range(5):
        paths = path + text_type[i] + '/'  # 경로 지정
        fileNameList = listdir(paths)  # 해당 경로에 포함된 파일 리스트를 모두 가져옴

        df = pd.DataFrame(fileNameList, columns=['file_name'])  # 파일 리스트 -> 데이터 프레임화
        df = df.astype('string')  # df type string 으로 변경

        cr = df['file_name'].str.contains('중등')  # 파일명에 '중등'이 포함되어있는 파일만 filtering
        data = df[cr]

        sub_df = get_data(data, paths)  # 데이터 가져오는 함수
        main_df = pd.concat([main_df, sub_df])  # main_labels에 누적하여 더하기

    return main_df

def word_tokenizing(txt, model = Komoran):

#    model = Okt
    temp = []
    temp.append(model.morphs(txt))

    return temp

def test_tokenizing(model):
    text = '안녕, 내 이름은 임가균이야.'
    tokenized_list = []
    tokenized_list.append(model.morphs(text))
    print(tokenized_list)
    return tokenized_list

#test_tokenizing(Komoran)


def tokenizing_data(model = Okt):

    main_df = load_data()
    main_df = main_df.reset_index(drop = True)
    list_split = []
    file_count = len(main_df)
    for i in range(file_count):
        list_split.append(main_df['essay'][i])

    tokenized_list = []
    max_length = 0
    for i in tqdm(range(len(list_split))):
        temp = list_split[i]
        tokenized_list.append(model.morphs(temp))
        if len(tokenized_list[i]) > max_length:
            max_length = len(tokenized_list[i])

    length_file_name = 'len_info.txt'
    with open(length_file_name, 'wt', encoding='utf-8') as myfile:
        myfile.write(str(max_length) + ' ' + str(file_count))

    return tokenized_list, max_length, file_count

def save_tokenized_data(list, file_name):
    file_name = file_name + '.prepro'
    with open(file_name, 'wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join([str(item) for item in list]))

    print(file_name + '파일 저장 완료')
    vec_data = word2vec.LineSentence(source=file_name)
    vec_model = word2vec.Word2Vec(list, vector_size=100, window=10, hs=1, min_count=1, sg=1)

#    vec_model_file = 'word2vec_model.model'
#    vec_model.save(vec_model_file)
#    print(vec_model_file + '모델 저장 완료')

    return vec_model

def save_word2vec_model(tokenized_list, file_name):
    vec_model = save_tokenized_data(tokenized_list, file_name)

    return vec_model

def load_word2vec_model(file_name, word2vec):
    model = word2vec.load(file_name)
    return model


def xtrain_padding(essay_file_count, essay_max_length, tokenized_list, word2vec):
    X_train = np.zeros((essay_file_count, essay_max_length, 100), dtype='f')

    for i in range(essay_file_count):
        for idx, word in enumerate(tokenized_list[i]):
            vector = word2vec.wv[word]
            X_train[i, idx, :] = vector

    print("패딩 완료")
    return X_train



tokenized_list, max_length, file_count = tokenizing_data(Okt)

test_list = word2vec.LineSentence('tokenized_with_Okt.prepro')
print(test_list)

vec_model = save_tokenized_data(tokenized_list, 'tokenized_with_Okt')

#word2vec_proto = save_word2vec_model(tokenized_list, 'tokenized_with_Okt')
#word2vec_model = load_word2vec_model('word2vec_model.model', word2vec)



len_file_name = 'len_info.txt'
with open(len_file_name, 'r', encoding='utf-8') as file:
    file_contents = file.read()

max_length, file_count = file_contents.split(" ")

max_length = int(max_length)
file_count = int(file_count)

print(file_contents)
print(max_length, file_count)

xtrain_padding(file_count, max_length, tokenized_list, vec_model)






#sns.set_theme(style='whitegrid', font_scale=1)  # seaborn style 지정
#sns.set_palette('Set2', n_colors=20)            # seaborn colar pallete 지정

#fig, ax = plt.subplots(1,2, figsize=(16,5))

#sns.histplot(main_df['scores'], ax=ax[0]);   # score 분포 확인
#sns.countplot(main_df['levels'], ax=ax[1]);  # levels 분포 확인

#plt.show()

