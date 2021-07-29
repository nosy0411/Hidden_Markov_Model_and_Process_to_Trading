# 관측 데이터 (Observations) 시퀀스를 이용하여 히든 상태의 시퀀스를 추정한다.
# 초기 확률, Transition 확률, Emission 확률이 주어졌을 때 Forward 알고리즘으로
# 관측 데이터가 발생할 확률을 계산한다.
#
# HMM 패키지는 hmmlearn을 사용한다.
# hmmlearn 설치 : conda install -c conda-forge hmmlearn
#     버전 확인 : conda list hmmlearn
#
# Name          Version        Build  Channel
# hmmlearn        0.2.1    py36h452e1ab_1000    conda-forge 
#
# ---------------------------------------------------------------------------
import numpy as np
from hmmlearn import hmm

# 히든 상태 정의
states = ["Rainy", "Sunny"]
nState = len(states)

# 관측 데이터 정의
observations = ["Walk", "Shop", "Clean"]
nObervation = len(observations)

# HMM 모델 빌드
model = hmm.MultinomialHMM(n_components=nState)
model.startprob_ = np.array([0.6, 0.4])
model.transmat_ = np.array([[0.7, 0.3], [0.4, 0.6]])
model.emissionprob_ = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])

# 관측 데이터 (Observations)
X = np.array([[0, 2, 1]]).T  # Walk, Clean, Shop

# Forwad(/Backward) algorithm으로 x가 관측될 likely probability 계산
logL = model.score(X)
p = np.exp(logL)
print("\nProbability of [Walk, Clean, Shop] = %.4f%s" % (p*100, '%'))
