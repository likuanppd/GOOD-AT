# GOOD-AT

This repo is for source code of ICLR 2024 paper "BOOSTING THE ADVERSARIAL ROBUSTNESS OF GRAPH NEURAL NETWORKS: AN OOD PERSPECTIVE".

Paper Link: https://openreview.net/forum?id=DCDT918ZkI&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2024%2FConference%2FAuthors%23your-submissions)

## Environment

- python == 3.8.8
- pytorch == 2.0.0+cu118
- scipy == 1.6.2
- numpy == 1.20.1
- deeprobust

## Test GOOD-AT on Perturbed Graphs
We provide an example of how to test GOOD-AT on a perturbed graph. We choose cora with 415 perturbations, which is 
generated from PGD in the unit test. If you want to conduct experiments on your own perturbed graphs, you just need to 
modify the dataset load module in this code. You can simply run

```python
python good.py
```

## Unit Test

Adversarial Unit Test is proposed by
Mujkanovic et al. (2022)[https://www.cs.cit.tum.de/daml/are-gnn-defenses-robust/]  which consists of seven adaptive attack methods.
Referring to (Günnemann, 2022), they categorize
the attack methods into seven classes and select the
most representative methods from each class as the
targets of adaptive attacks. The adversarial graphs
generated from these attacks can be encapsulated
together to test other defenses, and it can be regarded
as the minimum criterion for assessing the adaptive
robustness of the defense models. 

The unit_test module  used in this script is adapted from the following repository:
https://github.com/LoadingByte/are-gnn-defenses-robust

The `perturbed_graph/unit_test.npz` is the direct copy from https://github.com/harishgovardhandamodar/are-gnn-defenses-robust/blob/master/unit_test/unit_test.npz 

The gb module used in this script is also obtained from the same repository.

Our usage of the `unit_test` and `gb` modules from the referenced repository ensures comparable and scientific results in our experiments. By utilizing established modules from a reputable source, we maintain consistency with existing methodologies and contribute to the reproducibility of scientific findings in the field.

To conduct unit test for baselines, run 
```python
python baseline_unit_test.py
```
For GOOD-AT, run
```python
python goodat_unit_test.py
```

## Hyper-parameters
The hyper-parameters are well set in good.py and goodat_unit_test.py, and we do not modify it under any different perturbation rate.

- **K = 20:** the number of detectors
- **d_epochs = 50:** the number of epochs for training the detector
- **d_batchsize = 2048** the batch size for training the detector
- **ptb_d = 0.3:** the perturbation rate for training the detector
- **threshold = 0.1:** the threshold of detecting adversarial edges

## Self-training-based Poisoning defense
For the self-training defense, please refer to our another repository: https://github.com/likuanppd/STRG

## Citation
```
@inproceedings{li2023boosting,
  title={Boosting the Adversarial Robustness of Graph Neural Networks: An OOD Perspective},
  author={Li, Kuan and Chen, YiWen and Liu, Yang and Wang, Jin and He, Qing and Cheng, Minhao and Ao, Xiang},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```

## Contact

If you have any questions, please feel free to contact me with [likuan.ppd@gmail.com](mailto:likuan.ppd@gmail.com).

