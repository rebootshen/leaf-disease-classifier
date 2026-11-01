# apple-leaf-disease-classifier

Multi-label classification of apple leaf diseases for the Plant Pathology 2021 (FGVC8) Kaggle competition.

## Tech stack

- Python, Jupyter
- TensorFlow / Keras
- Xception, DenseNet121, EfficientNetB7, and a fused Xception + DenseNet121 network
- pandas, NumPy, scikit-learn, OpenCV, Matplotlib, Seaborn

## Architecture

The competition images are apple leaves labeled with one or more conditions: healthy, scab, rust, frog eye leaf spot, and complex. Training uses `ImageDataGenerator.flow_from_dataframe` so the full image set does not have to sit in memory. Images are resized before training. Pretrained ImageNet backbones are fine-tuned with early stopping, a learning-rate reduction on validation accuracy, and model checkpoints.

The notebook compares single backbones and a two-backbone model that concatenates Xception and DenseNet121 features. A separate submit-only notebook loads saved weights and writes `submission.csv`.

Charts from training and the Kaggle submission screenshot are in `asserts/` (the directory name is historical; it holds the project images).

## Layout

```text
PlantPathology-v3.ipynb          primary training notebook
notebook-train-*.ipynb           earlier training runs
notebook-submit-only-*.ipynb     inference and Kaggle submission
asserts/                         model diagrams, accuracy curves, submission screenshot
submission.csv                   sample submission
env.local.example                optional Kaggle API credentials
```

Start with `PlantPathology-v3.ipynb`. The other notebooks are earlier experiments.

## Data

- Original images: https://www.kaggle.com/c/plant-pathology-2021-fgvc8/data
- Resized images for local training: https://www.kaggle.com/ankursingh12/resized-plant2021

The image archive is not in this repository. Download it from Kaggle and point the notebook paths at the local copy. If you use the Kaggle CLI, put `KAGGLE_USERNAME` and `KAGGLE_KEY` in `env.local`.

## Run

Requirements: Python 3 with TensorFlow, Jupyter, and a GPU if you retrain. CPU inference is possible with the saved weights.

```bash
cp env.local.example env.local
# optional: fill KAGGLE_USERNAME and KAGGLE_KEY, then
jupyter notebook PlantPathology-v3.ipynb
```

Upload the best checkpoint to your own weight store before running a submit-only notebook. The notebooks expect those weights on the Kaggle input path or a local `model/` directory.

## What this demonstrates

- Transfer learning on a real multi-label image dataset
- A comparison of Xception, DenseNet121, EfficientNetB7, and a fused model
- Training callbacks, validation curves, and a Kaggle submission path separated from training

## Secrets

This repository has no API tokens. `env.local` is only for a local Kaggle API key and is gitignored.

## License

No license file is included yet. The Plant Pathology 2021 images remain under the competition's terms.
