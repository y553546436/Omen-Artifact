# Omen Test

### Workflow:

Train HDC model on datasets and output encoded test data to a directory.
```
python trainer.py [-h] --dataset DATASET [--dir DIR]

options:
  -h, --help         show this help message and exit
  --dataset DATASET  Dataset to train on
  --dir DIR          Directory to store encoded test data
```
e.g.
```
python trainer.py --dataset mnist
```

Run Omen partial inference on the encoded test data.
```
python partial.py [-h] [--dataset DATASET] [--data DATA] [--output OUTPUT]

Load and test model with Omen

options:
  -h, --help         show this help message and exit
  --dataset DATASET  dataset to use
  --data DATA        Directory containing model and test data
  --output OUTPUT    Directory to save output
```
e.g.
```
python partial.py --dataset mnist
```

Run performance estimation on the results from partial inference.
```
python estimate_perf.py --dataset isolet --strategy linear --start 64 --freq 64
```


### Environment Setup:

Install the required packages:
```
pip install -r requirements.txt
```

Note: You need to choose a python version that has pytorch pip package available. The current version that works is python<3.11.
