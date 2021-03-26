# Lint as: python3
r"""Code example for a custom model, using PyTorch.

This demo shows how to use a custom model with LIT, in just a few lines of code.
We'll use a transformers model, with a minimal amount of code to implement the
LIT API. Compared to models/glue_models.py, this has fewer features, but the
code is more readable.
This demo is similar in functionality to simple_tf2_demo.py, but uses PyTorch
instead of TensorFlow 2.
The transformers library can load weights from either,
so you can use any saved model compatible with the underlying model class
(AutoModelForSequenceClassification). To train something for this demo, you can:
- Use quickstart_sst_demo.py, and set --model_path to somewhere durable
- Or: Use tools/glue_trainer.py
- Or: Use any fine-tuning code that works with transformers, such as
https://github.com/huggingface/transformers#quick-tour-of-the-fine-tuningusage-scripts
To run locally:
  python -m lit_nlp.examples.simple_pytorch_demo \
      --port=5432 --model_path=/path/to/saved/model
Then navigate to localhost:5432 to access the demo UI.
NOTE: this demo still uses TensorFlow Datasets (which depends on TensorFlow) to
load the data. However, the output of glue.SST2Data is just NumPy arrays and
plain Python data, and you can easily replace this with a different library or
directly loading from CSV.
"""
import re
from typing import List

from absl import app
from absl import flags
from absl import logging
from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.app import JsonDict
from lit_nlp.lib import utils
import torch
import transformers

from utils_ner import get_labels
from utils_ner import I2b2Dataset

import argparse
import collections
import logging
import os
import random
import time
import numpy as np
import torch
from captum.attr import LayerIntegratedGradients, LayerDeepLift, NeuronDeepLift, LayerDeepLiftShap
from captum.attr import visualization as viz
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from termcolor import colored
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer

from utils_ner import get_labels, read_examples_from_file, convert_examples_to_features, predict, \
    predict_with_embeddings

# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_path", None,
    "Path to trained model, in standard transformers format, e.g. as "
    "saved by model.save_pretrained() and tokenizer.save_pretrained()")

flags.DEFINE_string(
    "labels", None,
    "Path to labels file")

flags.DEFINE_string(
    "test_data_dir", None,
    "Directory to data file where test.txt exists")


def _from_pretrained(cls, *args, **kw):
    """Load a transformers model in PyTorch, with fallback to TF2/Keras weights."""
    try:
        return cls.from_pretrained(*args, **kw)
    except OSError as e:
        logging.warning("Caught OSError loading model: %s", e)
        logging.warning(
            "Re-trying to convert from TensorFlow checkpoint (from_tf=True)")
        return cls.from_pretrained(*args, from_tf=True, **kw)


class NerModel(lit_model.Model):
    """Simple NER model."""

    compute_grads: bool = True  # if True, compute and return gradients.

    def __init__(self, model_name_or_path, labels_file='labels.txt'):
        self.LABELS = get_labels(labels_file)
        num_labels = len(self.LABELS)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path)
        model_config = transformers.AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            output_hidden_states=True,
            output_attentions=True,
        )
        # This is a just a regular PyTorch model.
        self.model = _from_pretrained(
            transformers.AutoModelForTokenClassification,
            model_name_or_path,
            config=model_config)
        self.model.load_state_dict(
            torch.load(os.path.join(model_name_or_path, 'pytorch_model.bin'), map_location='cpu'))
        self.model.eval()

    def max_minibatch_size(self):
        return 1

    def predict_minibatch(self,
                          inputs: List[JsonDict],
                          config=None) -> List[JsonDict]:

        # encoded_input = self.tokenizer.batch_encode_plus(
        #     [example["sentence"] for example in inputs],
        #     return_tensors="pt",
        #     add_special_tokens=True,
        #     max_length=128,
        #     pad_to_max_length=True)
        sentence = [seq['sentence'] for seq in inputs]
        tokens = ['[CLS]'] + self.tokenizer.tokenize(sentence[0]) + ['[SEP]']
        input_ids = [self.tokenizer.convert_tokens_to_ids(tokens)]
        input_mask = [[1] * len(input_ids[0])]
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        input_mask_tensor = torch.tensor(input_mask, dtype=torch.long)
        model_input = {"input_ids": input_ids_tensor,
                       "attention_mask": input_mask_tensor}
        model_output = self.model(**model_input)
        logits, embs, unused_attentions = model_output[:3]
        logits_ndarray = logits.detach().cpu().numpy()
        example_preds = np.argmax(logits_ndarray, axis=2)
        confidences = torch.softmax(torch.from_numpy(logits_ndarray), dim=2).detach().cpu().numpy()
        label_map = {i: label for i, label in enumerate(self.LABELS)}
        predictions = [label_map[pred] for pred in example_preds[0]]
        # ntok = torch.sum(input_mask_tensor, dim=1)
        outputs = {}
        for i, attention_layer in enumerate(unused_attentions):
            outputs[f'layer_{i}/attention'] = attention_layer[0].detach().cpu().numpy().copy()

        outputs['tokens'] = tokens
        outputs['bio_tags'] = predictions

        yield outputs

    def input_spec(self) -> lit_types.Spec:
        return {
            "sentence": lit_types.TextSegment()
        }

    def output_spec(self) -> lit_types.Spec:
        spec = {
            "tokens": lit_types.Tokens(),
            "bio_tags": lit_types.SequenceTags(align="tokens"),
        }
        for i in range(self.model.config.num_hidden_layers):
            spec[f'layer_{i}/attention'] = lit_types.AttentionHeads(align=("tokens", "tokens"))
        return spec


def main(_):
    # Load the model we defined above.
    models = {"NCBI BERT Finetuned": NerModel(FLAGS.model_path, labels_file=FLAGS.labels)}
    datasets = {"I2b2 2014": I2b2Dataset(data_dir=FLAGS.test_data_dir)}

    # Start the LIT server. See server_flags.py for server options.
    lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
    lit_demo.serve()


if __name__ == "__main__":
    app.run(main)
