import numpy as np

bh = np.array([[10, 6, 3], [5, 3, 15], [8, 9, 6]])
sp = np.array([[0.2, 0.5, 0.3], [0.8, 0.1, 0.1], [0, 1, 0]])
info = np.array([[0.7, 0.3, 0.1], [0.2, 0.6, 0.4], [0.1, 0.1, 0.5]])

pz = np.dot(sp, info.T)
print('p(z)=')
print(pz)
print()

pthz = np.zeros((3, 3, 3))

for i in range(3):
    for j in range(3):
        for k in range(3):
            pthz[i][j][k] = sp[i][k] * info[j][k] / pz[i][j]
    #         print(
    #             '{} * {}/ {} = {} '.format(sp[i][k], info[j][k], pz[i][j], pthz[i][j][k]))

    # print()

print('p(θ|z)=')
print(pthz)
print()


EU = np.zeros((3, 3, 3))
for i in range(3):
    EU[i] = np.dot(pthz[i], bh.T)

print('EU=')
print(EU)
print()

# print(np.max(EU, axis=2))
EU_max = np.diag(np.dot(pz, np.max(EU, axis=2).T))
EU_max_b = np.max(np.dot(sp, bh.T), axis=1)
print('max(EU) = ')
print(EU_max)
print('max(EU_before) =')
print(EU_max_b)
print('Value =')
print(EU_max - EU_max_b)

print(np.dot(sp, bh.T))
