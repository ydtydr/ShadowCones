# below we provide commands to train on different datasets, the following hyper-parameters could affect the performance
# umbral: radius in [0.05 (recommended), 0.1], margin in [0.0001, 0.001 (recommended), 0.01]
# penumbral: source=20.0 (recommended, height generally larger better), margin in [0.0001, 0.001 (recommended), 0.01]
# thus, you may need to varies the margin and radius/height hyper-parameters in your machine for a better result.

# Umbral-S-infinity, dataset: mammal
python train.py -debug 1 -dataset mammal -train_non_basic_percent 10 -dim 2 -model umbral -source infinity -radius 0.05 -margin 0.001 -epoch 400 -optimizer rsgd -lr 0.01 -eval_method partial
# Penumbral-S-infinity, dataset: mammal
python train.py -debug 1 -dataset mammal -train_non_basic_percent 10 -dim 2 -model penumbral -source 20.0 -margin 0.0001 -epoch 400 -optimizer rsgd -lr 0.01 -eval_method partial

# Umbral-S-infinity
# dataset: noun, MGC and hearst:
# use -margin 0.001 when train_non_basic_percent>0
python train_hogwild_lazy.py -num_processes 8 -debug 1 -dataset MCG -train_non_basic_percent 10 -dim 5 -model umbral -source infinity -radius 0.05 -margin 0.001 -epoch 400 -optimizer rsgd -lr 0.01 -eval_method partial
# it's recommended to use -margin 0.0001 when train_non_basic_percent=0
python train_hogwild_lazy.py -num_processes 8 -debug 1 -dataset MCG -train_non_basic_percent 0 -dim 5 -model umbral -source infinity -radius 0.05 -margin 0.0001 -epoch 400 -optimizer rsgd -lr 0.01 -eval_method partial

# Penumbral-S-infinity
# dataset: noun, MGC and hearst:
# use -margin 0.01 when train_non_basic_percent>0
python train_hogwild_lazy.py -num_processes 8 -debug 1 -dataset MCG -train_non_basic_percent 0 -dim 5 -model penumbral -source 20.0 -margin 0.01 -epoch 400 -optimizer rsgd -lr 0.01 -eval_method partial
# it's recommended to use -margin 0.0001 when train_non_basic_percent=0
python train_hogwild_lazy.py -num_processes 8 -debug 1 -dataset MCG -train_non_basic_percent 0 -dim 5 -model penumbral -source 20.0 -margin 0.0001 -epoch 400 -optimizer rsgd -lr 0.01 -eval_method partial