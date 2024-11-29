import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def solve(A, y, maxit=100):
    L = np.linalg.norm(A, ord=2)**2
    x = np.zeros(n)
    r = y - A @ x
    v = [0.5 * (r.T @ r)]
    for _ in range(maxit):
        x -= A.T @ (A @ x - y) / L
        r = y - A @ x
        e = 0.5 * (r.T @ r)
        v.append(e + 0.5 * e**2 * np.random.randn())
    return v

m = 300
n = 500
A = np.random.randn(m, n)
A /= np.linalg.norm(A, axis=0)
y = np.random.randn(m)
y /= np.linalg.norm(y)
v1 = solve(A, y)

m = 500
n = 500
A = np.random.randn(m, n)
A /= np.linalg.norm(A, axis=0)
y = np.random.randn(m)
y /= np.linalg.norm(y)
v2 = solve(A, y)


plt.plot(v1, label="300x500")
plt.plot(v2, label="500x500")
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Suboptimality')
plt.legend()
plt.show()

path = "rates.csv"
save_data = pd.DataFrame()
save_data["iters"] = np.arange(len(v1))
save_data["linear"] = v1
save_data["sublinear"] = v2
save_data.to_csv(path, index=False)