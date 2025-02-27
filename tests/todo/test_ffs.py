import reforge.forge.forcefields as ffs

ff = ffs.martini30rna()
# for key, value in ff.mapping.items():
#     print(key, value)

print(ff.name)
print(ff.sc_bonds('A'))
print(ff.sc_vs3s('A'))
print(ff.sc_atoms('A'))
