# 관측 데이터 (Observations) 시퀀스를 이용하여 히든 상태의 시퀀스를 추정한다.
# 초기 확률, Transition 확률, Emission 확률이 주어졌을 때 Viterbi 알고리즘으로
# 히든 상태의 시퀀스를 추정한다.
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
X = np.array([[0, 2, 1, 0]]).T

# Viterbi 알고리즘으로 히든 상태 시퀀스 추정 (Decode)
logprob, Z = model.decode(X, algorithm="viterbi")

# 결과 출력
print("\n  Obervation Sequence :", ", ".join(map(lambda x: observations[int(x)], X)))
print("Hidden State Sequence :", ", ".join(map(lambda x: states[int(x)], Z)))
print("Probability = %.6f" % np.exp(logprob))