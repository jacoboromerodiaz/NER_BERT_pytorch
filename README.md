# NER_BERT_pytorch
Application of BERT with PyTorch for NER using the few-nerd dataset.

## Overview

This project uses the BERT model for performing NER. BERT is a transformer-based model that has achieved state-of-the-art results in various NLP tasks. The pre-trained BERT model is fine-tuned on a NER dataset to recognize and classify entities.

This project is mainly based on Abishek Thakur's implementation, [seen here](https://github.com/abhishekkrthakur/bert-entity-extraction/tree/master)

## Requirements

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset used for this project was [few-nerd](https://huggingface.co/datasets/DFKI-SLT/few-nerd) from HuggingFace

## Usage
### Training & Evaluation

To train and evaluate the model, first modify the attributes of the ```config.py``` file specifying the model you wish to use, and in ```train.py``` the specific dataset in case another one is used (please note that the structure of the data must be coherent with the ```dataset.py```file).

Afterwards, run:

```bash
python train.py
```

### Inference

Before trying out the inference, make sure to change the sentence to make inference from in ``ìnference.py```.

Afterwards, run:

```bash
python inference.py
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## Authors
* [Jacobo Romero](https://github.com/jacoboromerodiaz/jacoboromerodiaz)
* [Abhishek Thakur](https://github.com/abhishekkrthakur)

