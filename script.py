import argparse
import collections
import logging
import os
import random

import numpy as np
import torch
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from termcolor import colored
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from utils_ner import get_labels, read_examples_from_file, convert_examples_to_features, predict

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Running NER explainer on {device}')
RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

parser = argparse.ArgumentParser()

parser.add_argument(
    "--tokenizer_name",
    default="",
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name",
)

parser.add_argument(
    "--model_type",
    default=None,
    type=str,
    required=True,
    help="Model type selected in the supported list",
)

parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    required=True,
    help="Path to pre-trained model or shortcut name selected in the supported list.",
)

parser.add_argument(
    "--cache_dir",
    default="",
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3",
)

parser.add_argument(
    "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
)

parser.add_argument(
    "--labels",
    default="",
    type=str,
    help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
)

parser.add_argument(
    "--max_seq_length",
    default=128,
    type=int,
    help="The maximum total input sequence length after tokenization. Sequences longer "
         "than this will be truncated, sequences shorter will be padded.",
)

parser.add_argument(
    "--data_dir",
    default=None,
    type=str,
    required=True,
    help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
)

parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

args = parser.parse_args()

TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

tokenizer_args = {k: v for k, v in vars(args).items() if v is not None and k in TOKENIZER_ARGS}
args.n_gpu = 0 if device == "cpu" else torch.cuda.device_count()
args.device = device
labels = get_labels(args.labels)
num_labels = len(labels)
label2id = {label: i for i, label in enumerate(labels)}
mode = "test"

tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    cache_dir=args.cache_dir if args.cache_dir else None,
    **tokenizer_args,
)

config = AutoConfig.from_pretrained(
    args.config_name if args.config_name else args.model_name_or_path,
    num_labels=num_labels,
    id2label={str(i): label for i, label in enumerate(labels)},
    label2id=label2id,
    cache_dir=args.cache_dir if args.cache_dir else None,
)
model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path)
model.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.bin'), map_location='cpu'))
model.to(device)

pad_token_label_id = CrossEntropyLoss().ignore_index
examples = read_examples_from_file(args.data_dir, mode)

features, all_tokens = convert_examples_to_features(
    examples,
    labels,
    args.max_seq_length,
    tokenizer,
    cls_token_at_end=bool(args.model_type in ["xlnet"]),
    cls_token=tokenizer.cls_token,
    cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
    sep_token=tokenizer.sep_token,
    sep_token_extra=bool(args.model_type in ["roberta"]),
    pad_on_left=bool(args.model_type in ["xlnet"]),
    pad_token=tokenizer.pad_token_id,
    pad_token_segment_id=tokenizer.pad_token_type_id,
    pad_token_label_id=pad_token_label_id,
)

all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

eval_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

prefix = ""
args.eval_batch_size = 1 * max(1, args.n_gpu)
# Note that DistributedSampler samples randomly
eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(
    eval_dataset)
eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

# multi-gpu evaluate
if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)

# Eval!
logger.info("***** Running evaluation %s *****", prefix)
logger.info("  Num examples = %d", len(eval_dataset))
logger.info("  Batch size = %d", args.eval_batch_size)
eval_loss = 0.0
nb_eval_steps = 0
preds = None
out_label_ids = None
model.eval()

explainer = LayerIntegratedGradients(predict, model.bert.embeddings)
example_index = 0

all_attributions = []

for batch in tqdm(eval_dataloader, desc="Evaluating"):
    example_attrs = []
    batch = tuple(t.to(args.device) for t in batch)

    input_ids = batch[0]
    attention_mask = batch[1]
    segment_ids = batch[2]
    batch_labels = batch[3]

    with torch.no_grad():
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}
        outputs = model(**inputs)
        tmp_eval_loss, logits = outputs[:2]

        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

        eval_loss += tmp_eval_loss.item()
    nb_eval_steps += 1
    logits_ndarray = logits.detach().cpu().numpy()
    if preds is None:
        preds = logits_ndarray
        out_label_ids = inputs["labels"].detach().cpu().numpy()
    else:
        preds = np.append(preds, logits_ndarray, axis=0)
        out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    # TODO To use DeepLIFT, ModelWrapper has to be created that overrides forward and possibly load_state_dict

    example_preds = np.argmax(logits_ndarray, axis=2)

    for token_index in np.where(example_preds[example_index] != 0)[0]:
        if out_label_ids[example_index][token_index] != pad_token_label_id:
            label_id = example_preds[example_index][token_index]
            target = (example_index, label_id)
            attributions, delta = explainer.attribute(input_ids, target=target,
                                                      additional_forward_args=(model, batch_labels),
                                                      return_convergence_delta=True)
            attributions = attributions.sum(dim=2).squeeze(0)
            attributions_sum = attributions / torch.norm(attributions)
            example_attrs.append({'attributions': attributions_sum, 'delta': delta,
                                  'token_index': token_index, 'label_id': label_id})

    all_attributions.append(example_attrs)

    example_index += 1

eval_loss = eval_loss / nb_eval_steps
confidences = torch.softmax(torch.from_numpy(preds), dim=2).detach().cpu().numpy()
preds = np.argmax(preds, axis=2)

label_map = {i: label for i, label in enumerate(labels)}

out_label_list = [[] for _ in range(out_label_ids.shape[0])]
predictions = [[] for _ in range(out_label_ids.shape[0])]

for i in range(out_label_ids.shape[0]):
    for j in range(out_label_ids.shape[1]):
        if out_label_ids[i, j] != pad_token_label_id:
            out_label_list[i].append(label_map[out_label_ids[i][j]])
            predictions[i].append(label_map[preds[i][j]])

results = {
    "loss": eval_loss,
    "precision": precision_score(out_label_list, predictions),
    "recall": recall_score(out_label_list, predictions),
    "f1": f1_score(out_label_list, predictions),
}

logger.info("***** Eval results %s *****", prefix)
for key in sorted(results.keys()):
    logger.info("  %s = %s", key, str(results[key]))

if mode == "test":
    print(classification_report(out_label_list, predictions))

# For printing the results ####
# TODO Needs to be accustomed for NER
index = 0
all_visualizations = []
for example in examples:

    print(colored(f'************ Example number {index + 1} ************', 'magenta'))

    colored_words = []

    for i, element in enumerate(example.words):
        if example.labels[i] != predictions[index][i]:
            colored_words.append(colored(element, 'red', attrs=['reverse', 'blink']))
            print(
                f'{element} was predicted as {predictions[index][i]} but it is actually {example.labels[i]}')
        elif example.labels[i] != 'O' and predictions[index][i] == example.labels[i]:
            colored_words.append(colored(element, 'green', attrs=['reverse']))
            print(f'{element} was correctly predicted as {predictions[index][i]}.')
        else:
            colored_words.append(colored(element, 'blue'))

    sentence = ' '.join(colored_words)
    print(sentence)

    example_attributions = all_attributions[index]

    for label_attribution in example_attributions:
        label_id = label_attribution['label_id']
        token_index = np.where(preds[index] == label_attribution['label_id'])[0][0]
        attributions_sum = label_attribution['attributions']
        delta = label_attribution['delta']
        score_vis = viz.VisualizationDataRecord(word_attributions=attributions_sum,
                                                pred_prob=max(confidences[index][token_index]),
                                                pred_class=labels[label_id],
                                                true_class=labels[label_id],
                                                attr_class=labels[label_id],
                                                attr_score=attributions_sum.sum(),
                                                raw_input=all_tokens[index],
                                                convergence_score=delta)
        all_visualizations.append(score_vis)

    index += 1

display = viz.visualize_text(all_visualizations)

with open('explanations.html', 'w') as file:
    file.write(display.data)
