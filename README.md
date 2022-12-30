# 상추의 생육 환경 생성 AI 경진대회


<div align=center>
  
  ![단락 텍스트](./img/Lettuce%20Growth%20Environment%20Prediction.png)
</div>


<div align="center">
    
  ![Python Version](https://img.shields.io/badge/Python-3.8.16-blue)
</div>


## Host
- 주최 : KIST 강릉분원
- 주관 : Dacon
- 기간 : 11월 21일 (월) 10:00 ~ 12월 19일 (월) 10:00
- https://dacon.io/competitions/official/236033/overview/description
---
## 🧐 About
생육 환경 생성 AI 모델 결과를 바탕으로 상추의 일별 최대 잎 중량을 도출할 수 있는 최적의 생육 환경 조성


1. 상추의 일별 잎중량을 예측하는 **AI 예측 모델** 개발 
2. 1번의 예측 모델을 활용하여 생육 환경 **생성 AI 모델** 개발 
  - 생성 AI 모델 결과로부터 상추의 일별 최대 잎 중량을 도출할 수 있는 최적의 생육 환경 조성 및 제안 

---
## 🖥️ Development Environment
```
Google Colab
OS: Ubuntu 18.04.6 LTS
CPU: Intel(R) Xeon(R) CPU @ 2.20GHz
RAM: 13GB
```
---
## 🔖 Project structure

```
Project_folder/
|- data/  # required data (csv)
|- feature/  # feature engineering (py)
|- garbage/  # garbage 
|- generative_model/  # CTGAN Model (pkl)
|- predict_model/  # Autogluon Model (pkl)
|- config  # Setting (py)
|- *model  # notebook (ipynb)
|_ [Dacon]상추의-생육-환경-생성-AI-경진대회_상추세요  # ppt (pdf) 
```
## 📖 Dataset
**Data Source**  [Train Test Dateset](https://dacon.io/competitions/official/236033/data)
```
Dataset Info.

- Input
Train, Test
CASE_01 ~ 28.csv (672, 16), TEST_01 ~ 05.csv (672, 16)
  DAT : 생육 일 (0~27일차)
  obs_time : 측정 시간
  상추 케이스 별 환경 데이터 (1시간 간격)

- Target
Train, Test
CASE_01 ~ 28.csv (28, 2), TEST_01 ~ 05.csv (28, 2)
  DAT : 생육 일 (1~28일차)
  predicted_weight_g : 일별 잎 중량
```


## 🔧 Feature Engineering
```
Feature selection.

누적값
- 구간별 시간에 대한 feature의 누적값
ex x 분무량
- 전체 평균에 대한 ec관측치와 분무량의 곱
수분량
- 자체 수분량 공식 사용
하루 평균
- 온도, 습도, co2, ec, 분무량, 적생광에 대한 하루 평균
Low-pass filter
- 누적값, ec x 분무량, 일평균에 적용
Kalman filter
- 누적값, ec x 분무량, 일평균에 적용
이동 평균
- 누적값, ec x 분무량, 수분량, 일평균에 적용
이동 중앙값
- 누적값, ec x 분무량, 수분량, 일평균에 적용
```

## 🎈 Modeling

**Predict Model**
```
AutoML: Autogluon, pycarat
Catboost
```
**Generative Model**
```
CTGAN
GAN
```


---
##  ✍️ Authors
- ``곽명빈`` [@ Myungbin](https://github.com/Myungbin?tab=repositories)
- ``김기범`` [@ 기범](https://github.com/gibum1228)
- ``반소희`` [@ sohi](https://github.com/BanSoHee)
- ``전주혁`` [@ jjuhyeok](https://github.com/jjuhyeok)
- ``최다희`` [@ Dahee Choi](https://github.com/daheeda)

---

## 😃 Result
- **Public score** 1st 3.16772 | **Private score** 4th 7.65751
- https://dacon.io/competitions/official/236033/overview/description

## 📖 Reference
CTGAN  
https://arxiv.org/abs/1907.00503  
https://github.com/sdv-dev/CTGAN  

생육환경  
https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=NPAP08069532&dbt=NPAP

## ❔Comments
[참여 후기](https://blog.naver.com/mbk1103_/222970298826)
