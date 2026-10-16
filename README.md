# CLFiLM-Net

Source code for **A closed-loop multimodal fusion network for hazardous materials transportation risk identification**.

CLFiLM-Net combines driver-monitoring images, vehicle trajectories and road-segment text to identify hazardous materials transportation risk events. An asymmetric closed-loop feature-wise modulation mechanism connects visual representations with the environmental representation derived from trajectories and text.

## Model

- **Visual encoder:** a ResNet-50 architecture with spatial attention, producing a 2,048-dimensional visual representation.
- **Trajectory encoder:** a two-layer bidirectional LSTM using normalized longitude, normalized latitude and cyclic hour features.
- **Text encoder:** `bert-base-chinese`, with the last two encoder layers and pooler trainable.
- **Environmental representation:** text-guided gating and bias combine the text and trajectory representations into a 128-dimensional vector.
- **Closed-loop fusion:** the environmental representation generates visual affine parameters; the modulated visual representation generates a residual feedback gate for the environmental representation. The two outputs are concatenated for classification.
- **Training objective:** multiclass focal loss with gamma set to 2.

The visual encoder is initialized from scratch. The text encoder loads pretrained BERT parameters. The FiLM parameter generator uses zero initialization in its final layer and a Tanh constraint on the multiplicative modulation parameter.

## Project files

| Path | Description |
|---|---|
| `train.py` | Dataset loading, model definitions, training, evaluation and plots |
| `vehicle_split.py` | Vehicle-level partitioning, overlap checks and split records |
| `requirements.txt` | Python dependencies |
| `tests/test_vehicle_split.py` | Synthetic tests for dataset partitioning |
| `.gitignore` | Git exclusions for local data, checkpoints and generated files |

The `RF_FiLM_ContextNet` class in `train.py` implements the CLFiLM-Net fusion network.

## Environment

Use a Python environment with compatible PyTorch and torchvision packages. A CUDA-enabled PyTorch installation is needed for GPU training.

```bash
python -m pip install -r requirements.txt
```

If the dependencies are already installed in a working environment, select that interpreter in PyCharm. The tokenizer and text encoder require access to `bert-base-chinese`, either through a download or an existing local cache.

## Data

The training script reads a prepared Excel or CSV table and the corresponding image directory. The table uses the following fields:

| Field | Meaning |
|---|---|
| `车牌号` | Vehicle identifier used for grouping and partitioning |
| `时间` | Observation timestamp |
| `经度` | Longitude |
| `纬度` | Latitude |
| `预警类型` | Risk-event class |
| `Image_Filename` | Image path relative to the image directory |
| `Location_Text` | Road-segment text |

The loader also accepts `具体路段` or `Unnamed: 10` as the road-text column name, with the eleventh column used as a fallback. Records are sorted by vehicle and time. Records lacking a risk-event label or road text are excluded.

Dataset reference: **危险品运输过程集成示范综合监控数据**, National Basic Science Data Center, CSTR `16666.11.nbsdc.wfvijrb9`.

- [Dataset record](https://www.nbsdc.cn/general/dataDetail?id=64c36e0c99f1de013a90358d&type=1)
- [CSTR identifier](https://cstr.cn/16666.11.nbsdc.wfvijrb9)

Obtain the data according to the provider's access conditions and prepare the table and matching images in the required format. The repository contains source code; data and trained checkpoints are not bundled.

## Dataset partitioning and inputs

Eligible vehicles have at least five records. Vehicles are assigned to training, validation and test partitions using seed 42 and target vehicle-count ratios of 60%, 20% and 20%. Sliding windows of length five are constructed separately within each partition. Vehicle identifiers do not overlap between partitions.

The ratios apply to vehicle counts. Window counts can have different proportions because vehicles have different numbers of records. At least five eligible vehicles are required to form three nonempty partitions.

Each window uses five trajectory observations, together with the image, road text and class label at the final observation. Images are resized to 224 by 224 pixels and normalized with mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. Longitude and latitude normalization uses statistics from the full input table. The class mapping is fitted to the labels in that table and shared across the three partitions.

`vehicle_split_record.json` records the seed, source-table SHA-256 digest, vehicle and window counts, class coverage and hashed vehicle identifiers. Review the class-coverage information when interpreting results, especially for rare classes.

## Training

Set the data paths in the main block near the bottom of `train.py`:

```python
CSV_FILE = r'C:\new_database\result\output\new_data666.xlsx'
IMG_DIR = r'C:\new_database\result\output\new'
```

In PyCharm, select the project interpreter and use **Run** on `train.py`. Set the working directory to the project folder. No command-line arguments are required. Alternatively, run the following command from that folder:

```bash
python train.py
```

| Setting | Value |
|---|---|
| Trajectory window length | 5 |
| Image size | 224 by 224 |
| Maximum text length | 32 tokens |
| Batch size | 10 |
| Epochs | 10 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Learning-rate scheduler | ReduceLROnPlateau on validation loss |
| Checkpoint selection | Highest validation macro-F1 |

The script selects CUDA when available and otherwise uses the CPU. A missing or unreadable image is represented by a zero tensor in the dataset loader.

## Evaluation and outputs

The training loop computes training, validation and test loss and accuracy each epoch, and uses validation macro-F1 for checkpoint selection. After training, the saved checkpoint is loaded for final test evaluation.

Outputs are written to the working directory:

- `vehicle_split_record.json`: partition metadata and class coverage.
- `base_tanh_best_weight.pth`: model state dictionary saved when validation macro-F1 improves.
- Per-class ROC/AUC and precision-recall figures.
- Training and test loss and accuracy curves.
- Console metrics: final test accuracy, macro-F1 and macro-recall.

Experimental metrics are obtained by running the training and evaluation procedure on the prepared data.

## Tests

```bash
python -m unittest discover -s tests -v
```

These tests use synthetic tables to check vehicle separation, complete assignment of eligible vehicles, deterministic partitioning, invalid partition rejection, window counts, shared class mapping, coordinate normalization and the separation of trajectory observations across partitions. They do not download BERT or perform model training.

## 中文运行说明

1. 在 PyCharm 中打开整个项目文件夹，并选择已经配置好的 Python 环境。
2. 在 `train.py` 中设置 `CSV_FILE` 和 `IMG_DIR`。
3. 将运行配置的工作目录设为项目文件夹，右键 `train.py` 并选择运行。
4. `vehicle_split.py` 由主程序自动调用，无需单独运行。
5. 训练权重、划分记录和评价图像会生成在工作目录中。
