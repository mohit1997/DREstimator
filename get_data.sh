yeast=https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data
ecoli=https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data
glass=https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data
vowel=https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/vowel/vowel-context.data

wget $yeast
wget $ecoli
wget $glass
wget -O vowel.data $vowel
