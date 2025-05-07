# Description
This repository contains a Jupyter Notebook with the code and description for a research on "Distillation of LLMs for the purpose of test assertion generation".

# How to run
The notebook can be ran by opening it using Anaconda 3.

To run the notebook, some training data from the teacher model is needed. To get this data, you must be able to run the teacher model, which is very computationally expensive. Therefore, the teacher model data was generated and provided from TU Delft. Assuming sufficient computational power, the data generation can be reproduced as follows:

1. The data that was used for the training of the student model originates from the following replication package: https://zenodo.org/records/14703162. 
2. The data from the replication package was used to fine-tune the teacher model (DeepSeek Coder 1.3B). 
3. After that, the data was ran again on the fine-tuned teacher model to generate the logits.
4. Due to the huge size of the output logits, the data was compressed in 4 bits. It is decompressed entry by entry when training the student model.
For more information (and an example of data generation from a CodeT5 teacher model), you can look at the /DataGeneration directory, and more specifically /DataGeneration/README.md. You will still need to create a /DataGeneration/data/assertion_dataset.jsonl file, which you can get from the replication package.

Once you have the generated data, you can put it in /Data/dataset_with_predictions.jsonl.
