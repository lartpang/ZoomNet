# (CVPR 2022) Zoom In and Out: A Mixed-scale Triplet Network for Camouflaged Object Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![GitHub last commit](https://img.shields.io/github/last-commit/lartpang/ZoomNet?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/lartpang/ZoomNet?style=flat-square)
![GitHub stars](https://img.shields.io/github/stars/lartpang/ZoomNet?style=flat-square)

## Changelog

* 2020/3/28: Initialize the repository.

## Usage

### Dependencies

Some core dependencies:

- timm == 0.4.12
- torch == 1.8.1
- [pysodmetrics](https://github.com/lartpang/PySODMetrics) == 1.2.4 # for evaluating results

More details can be found in <./requirements.txt>

### Training

You can use our default configuration, like this:

```shell
$ python main.py --model-name=ZoomNet --config=configs/zoomnet/zoomnet.py --datasets-info ./configs/_base_/dataset/dataset_configs.json --info demo
```

or use our launcher script to start the one command in `commands.txt` on GPU 1:

```shell
$ python tools/run_it.py --interpreter 'abs_path' --cmd-pool tools/commands.txt  --gpu-pool 1 --verbose --max-workers 1
```

If you want to launch multiple commands, you can use it like this:

1. Add your commands into the `tools/commands.txt`.
2. `python tools/run_it.py --interpreter 'abs_path' --cmd-pool tools/commands.txt --gpu-pool <gpu indices> --verbose --max-workers max_workers`

**NOTE**:

- `abs_path`: the absolute path of your python interpreter
- `max_workers`: the maximum number of tasks to start simultaneously.

### Testing

For ease of use, we create a `test.py` script and a use case in the form of a shell script `test.sh`.

```shell
$ sudo chmod +x ./test.sh
$ ./test.sh 0  # on gpu 0
```

### Method Comparisons

- PySODEvalToolkit: A Python-based Evaluation Toolbox for Salient Object Detection and Camouflaged Object Detection
    - <https://github.com/lartpang/PySODEvalToolkit>

## Paper Details

### Method Detials

![](./assets/feat.png)

![](./assets/net.png)

### Comparison

#### Camouflaged Object Detection

![](./assets/cod_vis.png)

![](./assets/cod_cmp.png)

![](./assets/cod_fmpr.png)

#### Salient Object Detection

![](./assets/sod_cmp.png)

![](./assets/sod_fmpr.png)
