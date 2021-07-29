# 히든 마코프 모델 (HMM)을 적용하기위해 최적 상태 개수를 결정한다.
# 로그 우도함수의 BIC가 최소가 되는 상태 개수를 찾는다.
#
# 출처 논문 : 
# [1] Hassan, Md. R.; Nath, B., 2005, Stock Market Forecasting Using Hidden Markov Models: A New approach
# [2] Nguyet Nguyen , 2016, Stock Price Prediction using Hidden Markov Model
#
# -------------------------------------------------------------------------------------------------------
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

closePriceOnly = 0
if closePriceOnly == 1:
    # 종가 (Close)만으로 분석한다
    dataset = np.array(df)[:,3]
    dataset = np.reshape(dataset, (-1, 1))
else:
    # OHLC로 분석한다.
    dataset = np.array(df)

# BIC를 이용하여 상태의 최적 개수를 결정한다.
# 논문 [2].2 Model Selection 참조
nIter = 1000

# hmmlearn에서는 T를 작게 잡으면 아래 에러가 발생할 수 있음. T를 늘려야 함.
# ValueError: rows of transmat_ must sum to 1.0
T = 200
plt.figure(figsize=(8, 5))

# 상태 갯수를 2-9 까지 변화해 가면서 확인해보자. 적합한 상태 갯수 결정하는데 ML,DL,RL 을 사용할 수도 있을듯하다.
for nState in range(2, 10):
    BIC = []
    nStart = 0
    nEnd = T
    
    # 처음 블록 (T 개)
    X = dataset[nStart:nEnd, :]
    model = hmm.GaussianHMM(n_components=nState, tol=0.001, n_iter=nIter, covariance_type="diag")   
    model = model.fit(X)
    
    L = model.score(X)          # log likelihood
    k = nState ** 2 + nState * 2 - 1
    M = X.shape[0] * X.shape[1]
    BIC.append(-2 * L + k * np.log(M))
    
    P = model.startprob_
    A = model.transmat_
    mu = model.means_
    covar = model.covars_
    
    if model.monitor_.iter == nIter:
        print("# 수렴하지 못했습니다. nIter = %d가 너무 작습니다" % nIter)
        
    # 두 번째 블록부터
    for nSimulations in range(2,301):
        nStart += 1
        nEnd += 1
        X = dataset[nStart:nEnd, :]
        model = hmm.GaussianHMM(n_components=nState, tol=0.001, n_iter=nIter, covariance_type="diag")
        model.startprob_ = P
        model.transmat_ = A
        model.means_ = mu
        model.covars_ = covar     
        model = model.fit(X)
        
        L = model.score(X)          # log likelihood
        k = nState ** 2 + nState * 2 - 1
        M = X.shape[0] * X.shape[1]
        BIC.append(-2 * L + k * np.log(M))
        
        P = model.startprob_
        A = model.transmat_
        mu = model.means_
        covar = model.covars_
        
        if model.monitor_.iter == nIter:
            print("# 수렴하지 못했습니다. nIter = %d가 너무 작습니다" % nIter)

    print("State : %d" % nState)
    plt.plot(BIC, label = 'State ' + str(nState))

plt.legend(loc = 'upper left', fontsize=10)
plt.show()

