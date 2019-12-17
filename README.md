# DREstimator

Project for course CS498RL offered by Prof. Nan Jiang at UIUC. The code is a reimplementation of the paper [Doubly Robust Policy Evaluation and Learning](https://arxiv.org/pdf/1103.4601.pdf).

## Requirements
1. Python3 (<= 3.6.8)
2. Numpy
3. Sklearn


### Download Repository
Download:
```bash
git clone https://github.com/mohit1997/DREstimator.git
```

### Download Datasets
```bash
bash get_data.sh
```

### Run Policy Evaluation
1. Choosing the dataset, edit following lines in file `policy_eval.py`
```python
param_file = 'glass.params' # this file would be generated
filepath = 'glass.data' # Can be replaced with vowel.data, yeast.data or ecoli.data
```
2. Run `python policy_eval.py`

### Run Policy Optimisation
1. Choosing the dataset, edit following lines in file `policy_opt.py`
```python
param_file = 'glass.params' # this file would be generated
filepath = 'glass.data' # Can be replaced with vowel.data, yeast.data or ecoli.data
```
2. Run `python policy_opt.py`

