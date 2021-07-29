# 장중 정보기반 거래자의 비중과, 거래 상태를 HMM 으로 추론한다.
# Hidden 상태는 2개로 설정한다. 정보기반 거래자 및 유동성 거래자 상태 관측 데이터는 
# 거래량 1,500개 발생 때마다 매수 수량과 매도 수량을 측정하고, 매수 강도를 측정한다.
#
# 매수 강도 = 매수 수량 / 매도 수량
# 매수 강도가 1 에 가까워지면 정보보유 거래자의 참여 비중이 낮은 것이고,
# 1보다 커질수록 호재성 정보보유 거래자 비중이 높은것이고,
# 1보다 작아질수록 악재성 정보보유 거래자 비중이 높은 것임.
#
# 분석 데이터 : Euro Currency Futures EUR/USD (2016.8.16 09:32:39 ~ 익일 01:08:00)
#
# ------------------------------------------------------------------------------
from hmmlearn import hmm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
import os
np.set_printoptions(precision=3)

# 관측 데이터를 읽어온다
PATH=os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(PATH+'/StockData/Volume_ECU16.csv', index_col=0, parse_dates=True)

# 매수, 매도 수량의 비율을 계산한다
data = np.array(df['buy/sell'])
data = np.reshape(data, [data.shape[0], 1])

nState = 4

# HMM 모델을 빌드한다
model = hmm.GaussianHMM(n_components=nState, tol=0.00001, n_iter=5000)
model.fit(data)
 
# 히든 상태를 추정한다
hState = model.predict(data)
 
# 추정된 히든 상태의 분포를 가져온다
mu = np.array(model.means_)
sigma = np.array([np.sqrt(x) for x in model.covars_])
sigma = np.reshape(sigma, [nState, 1])

# 추정된 히든 상태의 Transition 확률을 가져온다
P = np.array(model.transmat_)

# 결과를 확인한다
print("\nmu :\n", mu)
print("\nvol :\n", sigma)
print("\nTransition :\n", P)

# 히든 상태의 변화를 확인한다
plt.figure(figsize=(15, 6))
plt.plot(hState, 'r-')
plt.title("Change of hidden state", fontsize=15)
plt.grid(True, alpha=0.5)
plt.show()

# 매수 강도 차트를 그려본다
plt.figure(figsize=(15, 6))
plt.plot(data)
plt.title("Propotional of Buy/Sell", fontsize=15)
plt.axhline(y=1.0, color='red')
plt.grid(True, alpha=0.5)
plt.show()

# 히든 상태의 분포를 확인한다
plt.figure(figsize=(15, 6))
for i in range(nState):
    x = np.linspace(mu[i] - 4*sigma[i], mu[i] + 4*sigma[i], 100)
    plt.plot(x, stats.norm.pdf(x, mu[i], sigma[i]), linewidth=3, label = 'State : '+str(i))
plt.title("Hidden States (" + str(nState) + " states)", fontsize=15)
plt.axvline(x=1, color='r', linestyle='--')
plt.legend(loc = 'upper left', fontsize=15)
plt.grid(True)
plt.show()

# 각 State의 비중을 계산한다. 이 비중으로 PIN을 추정할 수 있다.
print()
for i in range(nState):
    prob = len(np.where(hState == i)[0]) / len(hState)
    print("State %d : %.2f %s" % (i, prob * 100, '%'))
    