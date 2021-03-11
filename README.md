# Stock Prediction project  :chart_with_upwards_trend:


본 프로젝트는 머신러닝과 딥러닝을 포함한 다양한 방법으로 주가를 예측하고자 진행한 프로젝트입니다.

현재까지 총 2가지의 미니 프로젝트로 구성되어 있으며, 그 내용은 다음과 같습니다.


### 1. 뉴스 데이터를 활용한 KOSPI 지수 예측
  - pandas_datareader 로 KOSPI 지수 데이터 수집
  - BeautifulSoup 로 Naver, Paxnet 사이트의 뉴스 데이터 수집
  - konlpy 로 뉴스 데이터의 형태소를 분석한 후, 감성 분석 모델 적용
  
### 2. 딥러닝을 활용한 종목별 주가 예측
  - pandas_datareader(get_data_yahoo) 로 주가 데이터 수집
  - tensorflow 의 LSTM 예측 모델 생성
  

--------------------

 
#### cf1) tensorflow 설치 과정

명령 프롬프트 (cmd) 창을 실행하여 conda update -n base conda 를 입력합니다.

tensorflow 는 Python 버전 3.5 ~ 3.7 에서만 설치가 가능하므로, conda create -n tensor_37_env python=3.7 로 3.7 버전 가상환경을 생성합니다.

그 후, 순서대로 conda activate tensor_37_env (-> 가상환경 이름) 와 conda install tensorflow 를 입력하여 설치합니다.


#### cf2) tensorflow 가상환경에서 jupyter notebook 실행 방법

명령 프롬프트 (cmd) 창을 실행하여 conda activate tensor_37_env 를 입력합니다.

처음 실행하는 경우에는 pip install jupyter notebook (pip install ipykernel) 를 입력하여 설치합니다.

python -m ipykernel install --user --name tensor_37_env --display-name "[Tensorflow_37_env]" 으로 가상환경에 커널을 연결합니다.

마지막으로 jupyter notebook 을 입력하여 실행합니다.
