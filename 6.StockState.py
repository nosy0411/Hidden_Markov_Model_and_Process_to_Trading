# 관측 데이터 (Observations) 시퀀스를 이용하여 Transition 확률과 히든 상태를 추정한다.
# 관측 데이터만 주어졌을 때 Baum Welch 알고리즘으로 나머지를 모두 추정한다.
#
# ---------------------------------------------------------------------------------
from hmmlearn import hmm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from MyUtil import YahooData
import scipy.stats as stats
import os
np.set_printoptions(precision=3)


# Yahoo site에서 코스피 지수 데이터를 읽어온다
PATH=os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(PATH+'/StockData/^KS11.csv', index_col=0, parse_dates=True)

# 종가를 기준으로 주간 수익률을 계산한다.
S = df['Close']
df['Rtn'] = np.log(S) - np.log(S.shift(5))
df = df.dropna()
data = np.array(df['Rtn'])
data = np.reshape(data, [data.shape[0], 1])

nState = 3
annualizeFactor = 52

# HMM 모델을 빌드한다
model = hmm.GaussianHMM(n_components=nState, tol=0.001, n_iter=5000)
model.fit(data)
 
# 히든 상태를 추정한다
hState = model.predict(data)
 
# 추정된 히든 상태의 분포를 가져온다
mu = np.array(model.means_) * annualizeFactor
sigma = np.array([np.sqrt(x) for x in model.covars_]) * np.sqrt(annualizeFactor)
sigma = np.reshape(sigma, [nState, 1])

# 추정된 히든 상태의 Transition 확률을 가져온다
P = np.array(model.transmat_)

# 결과를 확인한다
print("\nmu :\n", mu)
print("\nvol :\n", sigma)
print("\nTransition :\n", P)

# 히든 상태의 변화를 확인한다
plt.figure(figsize=(10, 5))
plt.plot(hState[-300:], 'r-')
plt.grid(True, alpha=0.5)
plt.show()

# 주가 차트를 그려본다
plt.figure(figsize=(10, 5))
plt.plot(S[-300:])
plt.grid(True, alpha=0.5)
plt.show()

# 히든 상태의 분포를 확인한다
plt.figure(figsize=(10, 5))
for i in range(nState):
    x = np.linspace(mu[i] - 4*sigma[i], mu[i] + 4*sigma[i], 100)
    plt.plot(x, stats.norm.pdf(x, mu[i], sigma[i]), linewidth=3, label = 'State : '+str(i))
plt.title("Hidden States (" + str(nState) + " states)", fontsize=15)
plt.axvline(x=0, color='r', linestyle='--')
plt.legend(loc = 'upper left', fontsize=15)
plt.grid(True)
plt.show()

# 각 State의 비중을 계산한다.
print()
for i in range(nState):
    prob = len(np.where(hState == i)[0]) / len(hState)
    print("State %d : %.2f %s" % (i, prob * 100, '%'))
    