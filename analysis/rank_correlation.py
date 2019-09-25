import scipy.stats

cola = [1, 2, 18, 9, 14, 19, 12, 3, 13, 8, 17, 10, 11, 4, 7, 15, 16, 5, 6, 20]
mrpc = [4, 8, 10, 11, 14, 15, 16, 18, 7, 1, 5, 9, 13, 17, 3, 6, 20, 12, 2, 19]

sst_full = [15, 17, 20, 18, 5, 3, 4, 6, 7, 8, 9, 19, 2, 13, 14, 1, 16, 11, 12, 10]
sst_5000 = [8, 16, 3, 7, 6, 11, 12, 20, 5, 15, 17, 14, 18, 2, 13, 9, 1, 4, 10, 19]

for rank in [cola, mrpc, sst_full, sst_5000]:
    print(len(rank))


print(scipy.stats.spearmanr(cola, mrpc))
print(scipy.stats.spearmanr(sst_full, mrpc))
print(scipy.stats.spearmanr(sst_full, cola))
#print(scipy.stats.spearmanr(sst_full, sst_5000))

