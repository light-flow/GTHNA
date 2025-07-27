# GTHNA: Local-global Graph Transformer with Memory Reconstruction for Holistic Node Anomaly Evaluation

## Abstract

Anomaly detection in graph-structured data is an inherently challenging problem, as it requires the identification of rare nodes that deviate from the majority in both their structural and behavioral characteristics. Existing methods, such as those based on graph convolutional networks (GCNs), often suffer from over-smoothing, which causes the learned node representations to become indistinguishable. Furthermore, graph reconstruction-based approaches are vulnerable to anomalous node interference during the reconstruction process, leading to inaccurate anomaly detection. In this work, we propose a novel and holistic anomaly evaluation framework that integrates three key components: a local-global Transformer encoder, a memory-guided reconstruction mechanism and a multi-scale representation matching strategy. These components work synergistically to enhance the model’s ability to capture both local and global structural dependencies, suppress the influence of anomalous nodes, and assess anomalies from multiple levels of granularity. Anomaly scores are computed by combining reconstruction errors and memory matching signals, resulting in a more robust evaluation. Extensive experiments on seven benchmark datasets demonstrate that our method outperforms existing state-of-the-art approaches, offering a comprehensive and generalizable solution for anomaly detection across various graph domains.

![framework](picture/framework.png)

## Usage

Firstly,  you need to install the required packages.

```python
pip install -r requirements.txt
```

You can choose which dataset to use, and run the command.

```
python main.py --dataset ${dataset}
```

