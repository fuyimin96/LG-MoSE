<p align="center">
  <h1 align="center">Language-Guided Mixture of Spectral Experts for Cross-Scenario Hyperspectral Anomaly Detection (TCSVT'26)</h1>
  <p align="center">
    <strong>Yimin Fu</strong></a>
    &nbsp;&nbsp;
    <strong>Peiyuan Ma</strong></a>
    &nbsp;&nbsp;
    <strong>Yinghao Xu</strong></a>
    &nbsp;&nbsp;
    <strong>Michael K. Ng</strong></a>
    &nbsp;&nbsp;
    <strong>Peng Ren</strong></a>
    &nbsp;&nbsp;
  </p>
  <br>

Pytorch implementation for ["**Language-Guided Mixture of Spectral Experts for Cross-Scenario Hyperspectral Anomaly Detection**"](https://ieeexplore.ieee.org/document/11638213)

> **Abstract:** *Hyperspectral anomaly detection (HAD) aims to identify atypical targets in hyperspectral images (HSIs) without requiring prior knowledge of anomaly characteristics. Existing HAD methods typically distinguish anomalies from background patterns in a self-supervised manner, where reconstruction errors are used as the anomaly indicator. However, their practical deployment is severely constrained by the reliance on the one-for-one detection scheme, which necessitates scenario-specific retraining for inference. Otherwise, the detection performance can severely degrade due to distribution shifts between training and testing scenarios. To address these limitations, we propose language-guided mixture of spectral experts~(LG-MoSE) for cross-scenario hyperspectral anomaly detection. Specifically, LG-MoSE leverages language priors to provide comprehensive spectral–spatial characterizations of background components, thereby preventing the reconstruction from collapsing into identical shortcut solutions based solely on spatial representations of single-image inputs. During encoding, diverse spectral representations corresponding to different spectral subspaces are adaptively aggregated under the guidance of scene-level textual descriptions. Then, the aggregated spectral representations are combined with spatial representations to enhance the understanding of the underlying background pattern. Moreover, a transposable state space block is embedded into each decoding stage to mitigate the over-dependence on specific structural regularities. Extensive experiments are conducted on eight commonly used HAD datasets, and the proposed method consistently achieves state-of-the-art performance across various scenarios and settings.*


## Requirements

To run this code, you'll need the following dependencies:

- Python 3.10
- Pytorch 2.6
- matplotlib
- numpy
- scipy
- torchvision

## Datasets
For the dataset used in this paper, please download the [datasets](http://xudongkang.weebly.com/data-sets.html) and move them into the corresponding subfolders (`./data`).

<p align="center">
    <img src=./had.png width="888">
</p>

After downloading the datasets, run the following command to conduct band selection:

```bash
python band_select.py
```


## Run the code
To run a specific method, set the `method` variable and execute the following commands:

```bash
method=LG-MoSE   # Change this to any method name (e.g., LREN, BockNet, OT-AD)
(cd "./methods/$method/" && python main.py) 
```

## Visualization Result
<p align="center">
    <img src=./vis_result.png width="888">
</p>


## Citation
If you find our work and this repository useful. Please consider giving a star :star: and citation.
```bibtex
@article{fu2026lgmose,
  title={Language-Guided Mixture of Spectral Experts for Cross-Scenario Hyperspectral Anomaly Detection},
  author={Fu, Yimin and Ma, Peiyuan and Xu, Yinghao and Ng, Michael K. and Ren, Peng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026},
  publisher={IEEE},
  doi={10.1109/TCSVT.2026.3719083}
}
```

## Thanks
We would like to thank [GT-HAD](https://github.com/NanWangAC/GT-HAD) and [MSNet](https://github.com/enter-i-username/MSNet) for providing useful references for our implementations of data loading and model training.