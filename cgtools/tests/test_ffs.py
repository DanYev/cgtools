import cgtools.cgtools.forcefields as ffs

ff = ffs.martini30rna()
# for key, value in ff.mapping.items():
#     print(key, value)

print(ff.name, ff.rnaBbBondDictC)
