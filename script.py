import argparse
import collections
import logging
import math
import pprint

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from termcolor import colored
from torch import nn, optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from autoencoder import EncoderRNN, DecoderRNN, train_autoencoder
from bert_explainer import BertExplainer
from utils_ner import get_labels
from utils_ner import read_examples_from_file, convert_examples_to_features

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

parser = argparse.ArgumentParser

parser.add_argument(
    "--tokenizer_name",
    default="",
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name",
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

args = parser.parse_args()

TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]
tokenizer_args = {k: v for k, v in vars(args).items() if v is not None and k in TOKENIZER_ARGS}

labels = get_labels(args.labels)
num_labels = len(labels)
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
    label2id={label: i for i, label in enumerate(labels)},
    cache_dir=args.cache_dir if args.cache_dir else None,
)
model = AutoModelForTokenClassification.from_pretrained(
    args.model_name_or_path,
    from_tf=bool(".ckpt" in args.model_name_or_path),
    config=config,
    cache_dir=args.cache_dir if args.cache_dir else None,
)
model.load_state_dict(torch.load(args.model_name_or_path, map_location='cpu'))
model.to(device)
print()

hidden_size = 384
encoder = EncoderRNN(384, config.hidden_size, hidden_size).to(device)
decoder = DecoderRNN(384, config.hidden_size, hidden_size).to(device)
encoder_optimizer = optim.Adam(encoder.parameters())
decoder_optimizer = optim.Adam(decoder.parameters())
criterion = nn.MSELoss()

pp = pprint.PrettyPrinter(indent=4)

pad_token_label_id = CrossEntropyLoss().ignore_index
examples = read_examples_from_file(args.data_dir, mode)

features = convert_examples_to_features(
    examples,
    labels,
    args.max_seq_len,
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
args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
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
for batch in tqdm(eval_dataloader, desc="Evaluating"):
    batch = tuple(t.to(args.device) for t in batch)

    input_ids = batch[0]
    attention_mask = batch[1]
    segment_ids = batch[2]
    batch_labels = batch[3]

    with torch.no_grad():
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if args.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if args.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don"t use segment_ids
        outputs = model(**inputs)
        tmp_eval_loss, logits = outputs[:2]

        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

        eval_loss += tmp_eval_loss.item()
    nb_eval_steps += 1
    if preds is None:
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()
    else:
        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
    explainer = BertExplainer(model)
    # TODO What actually goes into the explainer?
    relevance, attentions, self_attentions = explainer.explain(input_ids, segment_ids, attention_mask,
                                                               [o["span"] for o in out_label_ids.values()])

    input_tensor = torch.stack(
        [r.sum(-1).unsqueeze(-1) * explainer.layer_values_global["bert.encoder"]["input"][0] for r in
         relevance], 0)
    target_tensor = torch.stack(relevance, 0).sum(-1)
    encoder_loss = train_autoencoder(input_tensor, target_tensor, encoder,
                                     decoder, encoder_optimizer, decoder_optimizer, criterion,
                                     max_length=13)
    print('Encoder loss: %.4f' % encoder_loss)

eval_loss = eval_loss / nb_eval_steps
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
index = None
for example in examples:
    if index != example.example_id:
        pp.pprint(example.para_text)
        index = example.example_id
        print('\n')
        print(colored('***********Question and Answers *************', 'red'))

    ques_text = colored(example.question_text + " Unanswerable: " + str(example.unanswerable), 'blue')
    print(ques_text)
    prediction = colored(predictions[math.floor(example.unique_id / 9)][example]['text'], 'green',
                         attrs=['reverse', 'blink'])
    print(prediction)
    print('\n')
