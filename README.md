# Description
This repository contains a Jupyter Notebook with the code and description for a research on "Creating Local LLMs for Test Assertion Generation: A Comparative Study of Knowledge Distillation from CodeT5".

# How to run
The notebook can be ran by opening it using Anaconda 3.
If you want to run any of the separate python files, you can use Python 3.9 and create and activate a virtual environment, then install the requirements and run the chosen file:
```
# Windows
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Mac/Linux
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

# Get/replicate data
To run the notebook, some training data from the teacher model is needed. You can get it directly from https://zenodo.org/records/15711321 and then you can paste the two data files in /data_generation/data/codet5. 

If you instead want to replicate the data, this is what the /data_generation folder is for. To replicate this data, you must be able to run the teacher model, which is very computationally expensive. Therefore, the teacher model data was generated and provided from TU Delft. Assuming sufficient computational power, the data generation can be reproduced as follows:

1. The data that was used for the training of the student model originates from the following replication package: https://zenodo.org/records/14703162. 
2. The data from the replication package was used to fine-tune the teacher model (codet5-base). 
3. After that, the data was ran again on the fine-tuned teacher model to generate the logits.
4. Due to the huge size of the output logits, the data was compressed with LZ4. It is decompressed entry by entry when training the student model.

For more information (and an example of data generation from a CodeT5 teacher model), you can look at the /data_generation directory, and more specifically /data_generation/README.md. You will still need to add the original data from the replication package to generate the distillation data (and specify the path with --data_path as you can see in data_generation/train_codet5_assertions.py).

Once you have the generated data, you can put it in /data_generation/data/codet5/distillation_data_training.jsonl and /data_generation/data/codet5/distillation_data_validation.jsonl.

You can also find the final models from the notebook here: https://zenodo.org/records/15711321.

# File overview
distillation_pipeline.ipynb - This is the main distillation pipeline notebook where the results are generated and compared.
train_model.py - Contains a function to train a student model and a function to run that training given a model configuration.
/data_generation - Replicate the data generation as already described.
/data - Contains a student model dataset and a model to load the data into that dataset.
/evaluation - Contains evaluation functions to generate metrics for models.
/student_models - Describes the student model configurations and their output directories (the output can also be found at https://zenodo.org/records/15711321).
