import math
import collections
import json
import pprint
import torch
from torch import nn, optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader,
                              SequentialSampler,
                              TensorDataset)
import argparse

from tqdm import tqdm
from termcolor import colored
from utils_ner import read_examples_from_file, convert_examples_to_features, evaluate
from autoencoder import EncoderRNN, DecoderRNN, train_autoencoder

from bert_explainer import BertExplainer
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from utils_ner import get_labels

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

# input_data is a list of dictionary which has a paragraph and questions
with open("/content/drive/My Drive/train-v2.0.json") as f:
    squad = json.load(f)
    for article in squad["data"]:
        # input_data = []
        # i = 1
        for context_questions in article["paragraphs"]:

            input_data = []
            i = 1

            paragraphs = {"id": i, "text": context_questions["context"]}
            paragraphs["ques"] = [(x["question"], x["is_impossible"]) for x in context_questions["qas"]]
            input_data.append(paragraphs)
            i += 1

            pad_token_label_id = CrossEntropyLoss().ignore_index
            examples = read_examples_from_file(input_data)

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

            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

            result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test")

            # Run prediction for full data
            pred_sampler = SequentialSampler(dataset)
            pred_dataloader = DataLoader(dataset, sampler=pred_sampler, batch_size=9)

            predictions = []
            for input_ids, input_mask, segment_ids, example_indices in tqdm(pred_dataloader):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)

                # explainer = shap.DeepExplainer(model, [input_ids, segment_ids, input_mask])
                with torch.no_grad():
                    # tensor_output = model(input_ids, segment_ids, input_mask)
                    batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
                    # batch_start_logits, batch_end_logits = torch.split(tensor_output, int(tensor_output.shape[1]/2), dim=1)
                    # shap_values = explainer.shap_values([input_ids, segment_ids, input_mask])

                features = []
                examples_batch = []
                all_results = []

                print(len(examples), example_indices.max())
                for i, example_index in enumerate(example_indices):
                    start_logits = batch_start_logits[i].detach().cpu().tolist()
                    end_logits = batch_end_logits[i].detach().cpu().tolist()
                    feature = features[example_index.item()]
                    unique_id = int(feature.unique_id)
                    features.append(feature)
                    examples_batch.append(examples[example_index.item()])
                    all_results.append(RawResult(unique_id=unique_id,
                                                 start_logits=start_logits,
                                                 end_logits=end_logits))

                output_indices = predict(examples_batch, features, all_results, 30)
                predictions.append(output_indices)

                explainer = BertExplainer(model)
                relevance, attentions, self_attentions = explainer.explain(input_ids, segment_ids, input_mask,
                                                                           [o["span"] for o in output_indices.values()])
                input_tensor = torch.stack(
                    [r.sum(-1).unsqueeze(-1) * explainer.layer_values_global["bert.encoder"]["input"][0] for r in
                     relevance], 0)
                target_tensor = torch.stack(relevance, 0).sum(-1)
                loss = train_autoencoder(input_tensor, target_tensor, encoder,
                                         decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=13)

                print('Encoder loss: %.4f' % loss)

            # For printing the results ####
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
