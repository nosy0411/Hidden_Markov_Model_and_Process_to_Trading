# 히든 마코프 모델 (HMM)을 이용하여 코스피 지수를 예측한다.
#
# 출처 논문 : 
# [1] Hassan, Md. R.; Nath, B., 2005, Stock Market Forecasting Using Hidden Markov Models: A New approach
# [2] Nguyet Nguyen , 2016, Stock Price Prediction using Hidden Markov Model
#
# --------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
from MyUtil import YahooData
import os

# Yahoo site에서 코스피 지수 데이터를 읽어온다

PATH=os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(PATH+'/StockData/^KS11.csv', index_col=0, parse_dates=True)
df = df.drop('Volume', 1)
dataset = np.array(df)

nTest = 100
nInterval = 1   # 과거-K 시퀀스를 취할 때 이만큼 건너뜀.
K = 40          # 40일 (2개월) 시퀀스로 로그 우도를 계산함.
nIters=10000    # 반복 횟수
nState = 7      # 히든 상태 개수

predStock = []
nData = dataset.shape[0]
traceJ = []
for idx in range(nData-nTest, nData):
    trainD = dataset[0:idx,:]       # 0 ~ test data 위치 전까지
    testD = dataset[idx,:]          # test data 위치
    nTrain = trainD.shape[0]        # train data 개수
    
    if idx == nData-nTest:          # 처음이면
        model = hmm.GaussianHMM(n_components=nState, tol=0.001, n_iter=nIters)
    else:
        model = hmm.GaussianHMM(n_components=nState, tol=0.001, n_iter=nIters)
        model.transmat_ = transmat 
        model.startprob_ = startprob
        model.means_ = mu
        model.covars_ = cov

    model.fit(trainD) #X

    transmat = model.transmat_
    startprob = model.startprob_
    mu = model.means_
    cov = model.covars_

    # 현재 K-시퀀스의 로그 우도를 계산한다.
    lhEnd = idx
    lhStart = lhEnd - K
    currLH = model.score(trainD[lhStart:lhEnd, :]) #P값
    
    # 과거 K-시퀀스의 로그 우도를 계산한다.
    pastLH = []
    pastEnd = []     # pastLH를 계산한 시퀀스의 마지막 데이터 인덱스
    nPast = 40       # 과거 K-시퀀스를 nPast 개수 만큼만 확인한다.
    while True:
        lhEnd -= nInterval
        lhStart = lhEnd - K
        if (lhStart < 0) or (nPast <= 0):
            break
        pastLH.append(model.score(trainD[lhStart:lhEnd, :]))
        pastEnd.append(lhEnd - 1)   # pashLH의 마지막 주가 위치를 기록해 둔다
        nPast -= 1

    # 현재 우도와 과거 우도가 가장 유사한 것을 찾는다. j-번째
    j = np.argmin(np.absolute(pastLH - currLH))
    traceJ.append(j)
    
    # 주가를 예측한다.
    jPos = pastEnd[j]   # 최적 pashLH의 마지막 주가 위치
    rtn = (trainD[jPos+1,3] - trainD[jPos,3]) / trainD[jPos,3]  # 종가 수익률 계산
    predStock.append(dataset[idx,3] * (1 + rtn))    # 종가 예측
    print("idx = ", idx, "j = ", j, "jPos = ", jPos)

# 정확도를 계산한다
# 실제 주가와 예측 주가의 종가
actStock = dataset[nData - nTest:,3]
predStock = np.array(predStock)

# 실제 주가와 예측 주가의 diff 및 정확도 계산
aDiff = np.diff(actStock) > 0
pDiff = np.diff(predStock) > 0
accuracy = 100 * (aDiff.astype(int) == pDiff.astype(int)).sum() / nTest
print("\n* 정확도 = %.2f" % accuracy)

# 테스트 데이터의 주가 예측 결과를 그린다
plt.figure(figsize=(10, 5))
plt.plot(range(nTest), predStock,'r--o', label = 'Predicted price');
plt.plot(range(nTest), actStock,'b-o', label = 'Actual price')
plt.xlabel('Time steps')
plt.ylabel('Price')
plt.title('Kospi price (accuracy = ' + str.format("%.2f" % accuracy) +'%)', fontsize=15)
plt.grid(True)
plt.legend(loc = 'upper left')
plt.show()

