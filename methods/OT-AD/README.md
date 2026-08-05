# OT-AD: Optimal Transport-Guided Transformer for Hyperspectral Anomaly Detection

This paper is published on IEEE Transactions on Geoscience and Remote Sensing.

Abstract: Hyperspectral anomaly detection (HAD) is one of the research hotspots within the domain of remote sensing image interpretation. The key is how to effectively distinguish the anomaly from the background. Existing methods usually rely on the assumption that the background of hyperspectral images (HSIs) conforms to multivariate Gaussian distributions. However, for real data, this assumption does not necessarily fit. Building upon the study of background distributions in real hyperspectral data, this article leverages optimal transport theory to model the background reconstruction task. The optimal transport of reconstruction and background is mathematically proven to be converted into that of reconstruction and the original image. To facilitate the reconstruction process, the Transformer architecture is employed due to its inherent ability to compute similarity weights. Based on the above, the optimal transport-guided Transformer anomaly detector (OT-AD) is designed by applying the Wasserstein distance constraints derived from optimal transport theory to the deep network. The experimental validation conducted on multiple public datasets confirms the effectiveness and reliability of the proposed method.

![Flowchart of OT-AD](https://github.com/jiojio0505/OT-AD/blob/main/flowchart.png?raw=true)

You can use the following format for citation:

```bibtex  
@article{wang2026ot,
  title={OT-AD: Optimal Transport-Guided Transformer for Hyperspectral Anomaly Detection},
  author={Wang, Mengjiao and Li, Lingling and Jiao, Licheng and Liu, Xu and Liu, Fang and Chen, Puhua and Yang, Shuyuan},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026},
  volume={64},
  pages={1-18},
  publisher={IEEE},
  doi={10.1109/TGRS.2026.3655458}
}
```
