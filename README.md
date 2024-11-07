# Omen Artifact

This repository contains the artifact for the ASPLOS 2025 paper "Early Termination for Hyperdimensional Computing Using Inferential Statistics".
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

If you want to use CUDA for training, first install the NVIDIA Container Toolkit following [this instruction](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Remember to configure docker to use GPU following the "Configuring Docker" section in the web page. Then build the Docker image by running `docker build -f cuda.dockerfile -t artifact .` in the top directory of this repository.
Then run the following command to start the container:

```bash
docker run -it --rm -v .:/omen/workspace --gpus all artifact /bin/bash
```

## [Optional] Use Our Trained Models and Skip Training

To save the reviewers' time and avoid potential inconsistency of training results across different platforms, we provide the option of using our trained models. We uploaded the trained models to Google Drive. To use the uploaded models, download the zip file `all_models.zip` [here](https://drive.google.com/file/d/1ji3cbdqLh4uGsz0fReg1sh0deip7TMFn/view?usp=sharing) into the `src/` directory and unzip it under the `src/` directory of this repository `unzip -q all_models.zip`.
If you downloaded the trained models, the pipeline script will automatically use the downloaded models and skip the training process.

## Experiment Set 1: Table 6 [estimated ~1 hour if skip training, ~1 day if trained on CPU, and ~2 hours if trained on GPU]

### Collect the Main Accuracy and Runtime Data

Note that some runtime data in Table 6 are collected when running on a MicroController (MCU, see Section 7.1 "Execution Setup" in the paper) to verify the effectiveness of Omen in target edge scenarios, and other runtime data are collected on local machines. We provide our MCU code in `firmware/` directory for review, but we understand that the reviewers may not have the exact hardware to run the MCU code. Therefore, to save the reviewers' trouble to purchase and setup the exact MCU, we provide the options of (1) using our MCU data and local data, or (2) using our MCU data and collect your own local data.

#### [Option 1] Use Our MCU and Local Data

We provide our experiment data for MCU experiments in `mcu-output.zip` and for experiments on local machines in `local-output.zip`. To use these data, run the following command in the top directory of this repository:

```bash
unzip -q mcu-output.zip
unzip -q local-output.zip
```

#### [Option 2] Use Our MCU Experiment Data, Collect Local Data

Run the following command in the top directory of this repository. This command will (1) train the models (automatically skipped if you used our trained models) (2) generate inference C++ code for local experiments and (3) collect the runtime data on local machines:

```bash
python local_pipeline.py
```

### Collect the Accuracy Data for SV Baseline

The SV (Smaller Vector) baseline is specially handled because its parameter depends on the experiment data of Omen. To collect the accuracy data for the SV baseline, run the following command in the top directory of this repository. This command runs Python simulation to get the accuracy data for the SV baseline:

```bash
python smaller_vector_baseline.py
```

### Parse the Data and Generate the Table 6

With the data collected or provided by us, you can now parse the data and generate the latex/csv table for Table 6 by running the following command in the top directory of this repository:

```bash
python parse_results.py
```

Options:

- `--local`: Use all local data (enable this option if you are not using our MCU data); The script by default uses our MCU data and the local data (provided by us or collected by you).
- `--csv`: Generate csv table; The script by default generates a latex table.
The Table 6 in the paper uses our MCU and local data. Note that your collected runtimes may have small variations from our runtimes due to randomness, but you should see similar trends in the table.

## Experiment Set 2: Table 7 [estimated 1 minute]

The data for this experiment was collected by us in MCU with the script `breakdown_pipeline.py`, but because the reviewers may not have access to the exact MCU, we provide our data in `breakdown_ucihar_OnlineHD_binary_linear_s512_f64_a005.csv`.
To generate the latex table from the data, run the following command in the top directory of this repository:

```bash
python print_latency_breakdown_table.py
```

## Experiment Set 3: Table 8 [estimated <5 minutes]

This part of the experiment uses Python simulation to get the accuracy and dimension reduction data in presence of hardware noise without running the C++ code.
To run the experiment, run the following command in the top directory of this repository:

```bash
python error_bsc_ldc.py
```

This script collects the accuracy and dimension reduction data by simulation and saves them in `output/` directory. Then, you can parse the data and generate the latex table for Table 8 by running the following command in the top directory of this repository:

```bash
python parse_ber_table.py
```
