# ECS 170 – Stage 3: CNN Image Classification (MNIST / ORL / CIFAR)

Built on the course code template (Stages 1–2): `dataset → method → result → evaluate`, glued by a `setting`.

```
code/
  base_class/            dataset, method, result, evaluate, setting  (template abstract classes)
  stage_3_code/
    Dataset_Loader.py      loads the pickled MNIST / ORL / CIFAR files → (N, C, H, W) arrays
    Method_CNN.py          configurable CNN (inherits method + nn.Module): train / test / run
    Method_CNN_MNIST.py    MNIST architecture + hyper-parameters
    Method_CNN_ORL.py      ORL architecture + hyper-parameters (1 input channel)
    Method_CNN_CIFAR.py    CIFAR architecture + hyper-parameters (+ flip/crop augmentation)
    Evaluate_Metrics.py    accuracy + weighted/macro precision, recall, F1
    Result_Saver.py        pickles predictions + training history
    Setting_Train_Test.py  uses the provided train/test split (no CV needed)
    Plot_Learning_Curve.py loss + accuracy convergence plots
script/stage_3_script/
  script_cnn_mnist.py  script_cnn_orl.py  script_cnn_cifar.py   ← 3-3 / 3-4
  script_cnn_ablation.py                                        ← 3-5
data/stage_3_data/     ← put the three provided files here, named exactly MNIST, ORL, CIFAR
result/stage_3_result/ ← learning curves (.png), scores (.json), ablation tables (.csv/.md)
```

## Setup
1. Unzip the instructor's Stage 3 dataset and copy `MNIST`, `ORL`, `CIFAR` (no extension) into `data/stage_3_data/`.
2. `pip install torch scikit-learn matplotlib numpy` (or install them into the PyCharm `.venv`).
3. In PyCharm: mark the project root as the content root, set the working directory to `script/stage_3_script`, and run a script.
   From a terminal: `cd script/stage_3_script && python script_cnn_mnist.py`

## Running
| Step | Command | CPU time (approx.) |
|---|---|---|
| 3-3 MNIST | `python script_cnn_mnist.py` | ~5–10 min |
| 3-4 ORL | `python script_cnn_orl.py` | ~1–2 min |
| 3-4 CIFAR | `python script_cnn_cifar.py` | ~25–40 min (a few minutes on GPU) |
| 3-5 study | `python script_cnn_ablation.py ORL` | ~20 min |
| 3-5 study | `python script_cnn_ablation.py MNIST --epochs 3` | ~30 min |
| 3-5 study | `python script_cnn_ablation.py CIFAR --epochs 5 --only baseline depth_shallow depth_deeper kernel_large pool_avg stride2_no_pool hidden_large loss_mse optim_sgd no_batchnorm` | ~1 hr |

GPU (3-7): `Method_CNN.get_device()` automatically uses CUDA (NVIDIA) or MPS (Apple silicon) when available.

Every script prints accuracy and weighted/macro precision, recall and F1, then saves
`result/stage_3_result/CNN_<DATASET>_learning_curve.png` and `CNN_<DATASET>_scores.json`.
The ablation script writes `ablation_<DATASET>.md`, a table you can paste into the report, plus `ablation_<DATASET>_curves.png`.

## Data format assumed
Each file is a pickled dict `{'train': [{'image': ndarray, 'label': int}, ...], 'test': [...]}`.
MNIST images are 28×28; ORL images are 112×92×3 with R=G=B, so the loader keeps one channel, and labels 1–40 are remapped to 0–39; CIFAR images are 32×32×3.
If your files differ, only `Dataset_Loader._to_array` needs to change.
