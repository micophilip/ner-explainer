<p align="center">
  <a href="https://github.com/micophilip/ner-explainer/blob/master/LICENSE" alt="License">
    <img src=https://img.shields.io/apm/l/vim-mode?style=flat-square/>
  </a>
</p>

# Named Entity Recognition Explainer

This project adapts [Integrated Gradients](http://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf) implementation provided by Captum for named entity recognition task to explain
BERT model predictions on I2B2 2014 - PHI dataset. It also extends Language Interpretability Tool (LIT) to visualize
and debug NER BERT models. This project can explain any BERT-based model on NER task for any dataset in CONLL format.

## Acknowledgments

Code in this project was based on code in [HuggingFace](https://github.com/huggingface/transformers) and the extensive
examples provided by [Captum](https://captum.ai) and [LIT](https://github.com/PAIR-code/lit) repositories.

## Running

### Perequisites

* Any BERT model trained on NER for a CONLL-based dataset. The model needs to be trained by HuggingFace NER
[script](https://bit.ly/3dByYL0).
* A dataset in CONLL format. Test dataset in data folder assumed to be called `test.txt`
* Python 3.x (tested with 3.7)  
* Run `pip install -r requirements.txt`

### Captum

```
python explainer.py --data_dir /path/to/data/folder --model_type bert \ 
--labels /path/to/labels.txt --model_name_or_path /path/to/trained/model \
--max_seq_length 128 --explanations_dir /path/to/store/explainations.html
```
### Lit Server

```
python lit.py --model_path /pth/to/trained/model --labels /path/to/labels.txt
--test_data_dir /path/to/test/data/folder
```

## Dataset

The dataset used in this project is I2B2 2014 PHI dataset. Can be requested from the 
[Department of Biomedical Informatics](https://portal.dbmi.hms.harvard.edu) and is provided for free to students
and researchers. Any NER-annotated CONLL dataset can be used with this project.

## Results

Explanation results are stored in the explanations folder provided in an explanations.html file.
