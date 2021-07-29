# Observation 시퀀스만을 이용하여, 초기 확률, Transition, Emmision 확률을 
# 추정하고, 히든 상태 시퀀스를 추정한다. (Baum Welch 알고리즘)
#
# ----------------------------------------------------------------------
import numpy as np
from hmmlearn import hmm
np.set_printoptions(precision=2)

nState = 2
pStart = [0.6, 0.4]
pTran = [[0.7, 0.3], [0.2, 0.8]]
pEmit = [[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]]

# 1. 주어진 확률 분포대로 관측 데이터 시퀀스를 생성한다.
# ---------------------------------------------------
# 히든 상태 선택. 확률 = [0.6, 0.4]
s = np.argmax(np.random.multinomial(1, pStart, size=1))

X = []      # Obervation 시퀀스
Z = []      # 히든 상태 시퀀스
for i in range(5000):
    # Walk, Shop, Clean ?
    a = np.argmax(np.random.multinomial(1, pEmit[s], size=1))
    X.append(a)
    Z.append(s)
    
    # 히든 상태 천이
    s = np.argmax(np.random.multinomial(1, pTran[s], size=1))

X = np.array(X)
X = np.reshape(X, [len(X), 1])
Z = np.array(Z)

# 2. Observation 시퀀스만을 이용하여, 초기 확률, Transition, Emmision 확률을 추정하고,
#    히든 상태 시퀀스를 추정한다. (Baum Welch 알고리즘)
# EM 알고리즘은 local optimum에 빠질 수 있으므로, 5번 반복하여 로그 우도값이 가장
# 작은 결과를 채택한다.
# ---------------------------------------------------------------------------------
zHat = np.zeros(len(Z))
minprob = 999999999
for k in range(5):
    model = hmm.MultinomialHMM(n_components=nState, tol=0.0001, n_iter=10000)
    model = model.fit(X)
    predZ = model.predict(X)
    logprob = -model.score(X)
    
    if logprob < minprob:
        zHat = predZ
        T = model.transmat_
        E = model.emissionprob_
        minprob = logprob
    print("k = %d, logprob = %.2f" % (k, logprob))

# 3. 1 단계에서 생성한 Z와 추정한 zHat의 정확도를 측정한다.
# --------------------------------------------------------
accuracy = (Z == zHat).sum() / len(Z)
if accuracy < 0.5:
    T = np.fliplr(np.flipud(T))
    E = np.flipud(E)
    zHat = 1 - zHat
    print("flipped")

accuracy = (Z == zHat).sum() / len(Z)
print("\naccuracy = %.2f %s" % (accuracy * 100, '%'))

# 추정 결과를 출력한다
print("\nlog prob = %.2f" % minprob)
print("\nstart prob :\n", model.startprob_)
print("\ntrans prob :\n",T)
print("\nemiss prob :\n", E)
print("\niteration = ", model.monitor_.iter)
