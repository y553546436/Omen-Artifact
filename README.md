# Omen Artifact

This repository contains the artifact for the paper "Early Termination for Hyperdimensional Computing Using Inferential Statistics".
The artifact includes code and scripts to reproduce the experiments and tables in the paper. The experiments are divided into three sets, each corresponding to one table in the paper. Please run the experiments in the order of Experiment Set 1, Experiment Set 2, and Experiment Set 3, due to the dependency of the experiments (e.g., Experiment Set 2 and Experiment Set 3 require the trained models from Experiment Set 1).

## Requirements
We provide the Docker configuration file for the artifact.
Docker installation instructions can be found [here](https://docs.docker.com/get-docker/).
### [Option 1] CPU-only Docker Image
To build the Docker image, run `docker build -t artifact .` in the top directory of this repository.
Then run the following command to start the container:
```bash
docker run -it --rm -v .:/omen/workspace artifact /bin/bash
```

### [Option 2] CUDA-enabled Docker Image
If you want to use CUDA for training, first install the NVIDIA Container Toolkit following [this instruction](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). [configure docker to use gpu].Then build the Docker image by running `docker build -f cuda.dockerfile -t artifact .` in the top directory of this repository.
Then run the following command to start the container:
```bash
docker run -it --rm -v .:/omen/workspace --gpus all artifact /bin/bash
```

## [Optional] Skip Training the Models
To save the reviewers' time and avoid potential inconsistency of training results across different platforms, we provide the option of using our trained models. We uploaded the trained models to Google Drive. To use the uploaded models, download the zip file `all_models.zip` [here](https://drive.google.com/file/d/1ji3cbdqLh4uGsz0fReg1sh0deip7TMFn/view?usp=sharing) into the `src/` directory and unzip it under the `src/` directory of this repository `unzip -q all_models.zip`.
If you downloaded the trained models, the pipeline script will automatically use the downloaded models and skip the training process.

## Experiment Set 1: Table 6 [estimated xx~xx hours if skip training, xx~xx hours if trained on CPU, and xx~xx hours if trained on GPU]

### Collect the Main Accuracy and Runtime Data
Note that some runtime data in Table 6 are collected when running on a MicroController (MCU, see Section 7.1 "Execution Setup" in the paper) to verify the effectiveness of Omen in target edge scenarios. To save the reviewers' trouble to purchase and setup the exact MCU, we provide the option of using our experiment data.

#### [Option 1] Use Our Experiment Data
We provide our experiment data for MCU experiments in `mcu-output.zip` and for experiments on local machines in `local-output.zip`. To use these data, run the following command in the top directory of this repository:
```bash
unzip -q mcu-output.zip
unzip -q local-output.zip
```

#### [Option 2] Use Our MCU Experiment Data, Run the Local Experiments on Your Own
Run the following command in the top directory of this repository:
```bash
python local_pipeline.py
```

### Collect the Accuracy Data for SV Baseline
The SV (Smaller Vector) baseline is specially handled because its parameter depends on the experiment data of Omen. To collect the accuracy data for the SV baseline, run the following command in the top directory of this repository:
```bash
python smaller_vector_baseline.py
```

### Parse the Data and Generate the Table 6
With the data collected, run the following command in the top directory of this repository:
```bash
python parse_results.py
```

## Experiment Set 2: Table 7 [estimated xx~xx hours]

## Experiment Set 3: Table 8 [estimated xx~xx hours]