# Anomaly Transformer in PyTorch

This is a reimplementation of [**Anomaly Transformer**: Time Series Anomaly Detection with Association Discrepancy. (ICLR 2022)](https://openreview.net/forum?id=LzQQ89U1qm_) in PyTorch.
[Original code](https://github.com/thuml/Anomaly-Transformer) was stripped-down and rewrote (where necessary) to support my own research.


## Requirements

- Python 3.8 or later
- PyTorch 1.10 or later
- Numpy
- Pandas
- scikit-learn
- tqdm

## Usage

### Training

Use the following command to train an AT model on time series data.

```sh
python main.py --train --data data/train.csv \
    --n_features 21 --window_size 100
```

By default, the trained model is saved into the `models` folder.

### Testing

Use the following command to run the model on test data.

```sh
python main.py --test --data data/test.csv \
    --n_features 21 --window_size 100
```

AT outputs anomaly score per time step and the results are saved into a `csv` file.

Refer to the [`main.py`](main.py) file for a complete list of available command-line arguments.

## Data

The `data` folder contains two example datasets but is not provided here:

- `SWaT_Dataset_Normal_v0.csv`: The normal dataset used for training the model.
- `SWaT_Dataset_Attack_v0.csv`: The attack dataset used for testing the model.

These datasets are from the Secure Water Treatment (SWaT) testbed and contain time-series data from various sensors in a water treatment facility.

**To use the dataset, please request access from [iTrust's website](https://itrust.sutd.edu.sg/itrust-labs_datasets/).**

## Acknowledgements

Thanks for THUML's [original codebase](https://github.com/thuml/Anomaly-Transformer).

## Citation

If you find this implementation useful in your research, please consider citing the original paper:

```bibtex
@inproceedings{xu2022anomaly,
	title        = {Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy},
	author       = {Jiehui Xu and Haixu Wu and Jianmin Wang and Mingsheng Long},
	year         = 2022,
	booktitle    = {International Conference on Learning Representations},
	url          = {https://openreview.net/forum?id=LzQQ89U1qm_}
}
```
