import collections
import torch
import torch.nn.functional as F


class BertExplainer:
    def __init__(self, model):
        self.model = model
        self.layer_values_global = collections.OrderedDict()

    @staticmethod
    def save_inputs(self, layer_values, module_name):
        def _save_inputs(module, input, output):
            layer_values[module_name] = {"input": input, "output": output}

        return _save_inputs

    def register_hooks(self, model, layer_values, parent_name=""):
        modules = list(model.named_modules())
        for i, (name, module) in enumerate(modules):
            if not name:
                name = 'root'
            module.register_forward_hook(self.save_inputs(self, layer_values, name))
            # module.register_backward_hook(compute_relevance(layer_values, name))

        return layer_values

    @staticmethod
    def compute_matmul_relevance(self, input1, input2, relevance):
        input1_relevance = torch.zeros(input1.shape, dtype=torch.float, device="cpu")
        input2_relevance = torch.zeros(input2.shape, dtype=torch.float, device="cpu")
        for i in range(input1.shape[0]):
            for j in range(input1.shape[1]):
                out = torch.matmul(input1[i][j], input2[i][j])
                presum = input1[i][j].unsqueeze(-1) * input2[i][j]
                contributions = relevance[i][j].unsqueeze(-2) * presum / out.unsqueeze(-2)
                input1_relevance[i][j] = contributions.sum(-1)
                input2_relevance[i][j] = contributions.sum(0)
        # print("input1_relevance "+str(input1_relevance.sum()))
        # print("input2_relevance "+str(input2_relevance.sum()))

        return input1_relevance, input2_relevance

    @staticmethod
    def compute_input_relevance(self, weight, input, relevance):
        # print("compute_input_relevance begin")
        input_relevance = torch.zeros(input.shape, dtype=torch.float, device="cpu")
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                input_contributions = weight * input[i][j].unsqueeze(1)
                input_relevance[i][j] = (input_contributions * relevance[i][j].unsqueeze(0)).sum(-1)
                # print(input_relevance.sum())
        # print("input_relevance "+str(input_relevance.sum()))
        input_relevance_normalized = input_relevance / input_relevance.sum((-2, -1)).unsqueeze(-1).unsqueeze(-1)
        # print("input_relevance_normalized "+str(input_relevance_normalized.sum()))
        # print("compute_input_relevance end")

        return input_relevance_normalized

    def compute_linear_3(self, module, module_name, relevance):
        input = self.layer_values_global[module_name]["input"][0]
        weight = module.weight.transpose(-2, -1)
        output = torch.matmul(input, weight) + module.bias
        numerator = (output - module.bias)
        division = numerator / output
        relevance *= division
        input_relevance = self.compute_input_relevance(self, weight, input, relevance)

        return input_relevance

    def compute_bert_output(self, module, module_name, relevance):
        _, input_tensor = self.layer_values_global[module_name]["input"]
        hidden_states = self.layer_values_global[module_name + ".dropout"]["input"][0]
        layer_norm_relevance = relevance
        hidden_states_relevance = layer_norm_relevance.clone().detach()
        input_tensor_relevance = layer_norm_relevance.clone().detach()
        hidden_states_relevance = self.compute_linear_3(module.dense, module_name + ".dense", hidden_states_relevance)

        return hidden_states_relevance, input_tensor_relevance

    def compute_self_attention(self, module, module_name, relevance):
        relevance = module.transpose_for_scores(relevance)
        attention_probs = self.layer_values_global[module_name + ".dropout"]["input"][0]
        value_layer = module.transpose_for_scores(self.layer_values_global[module_name + ".value"]["output"])
        attention_probs_relevance, value_layer_relevance = self.compute_matmul_relevance(self, attention_probs,
                                                                                         value_layer,
                                                                                         relevance)

        query_layer = module.transpose_for_scores(self.layer_values_global[module_name + ".query"]["output"])
        key_layer = module.transpose_for_scores(self.layer_values_global[module_name + ".key"]["output"])
        query_layer_relevance, key_layer_relevance = self.compute_matmul_relevance(self, query_layer,
                                                                                   key_layer.transpose(-1, -2),
                                                                                   attention_probs_relevance)
        key_layer_relevance = key_layer_relevance.transpose(-1, -2)

        new_shape = (query_layer_relevance.size()[0], query_layer_relevance.size()[2]) + (module.all_head_size,)
        query_layer_relevance = query_layer_relevance.permute(0, 2, 1, 3).contiguous().view(new_shape)
        key_layer_relevance = key_layer_relevance.permute(0, 2, 1, 3).contiguous().view(new_shape)
        value_layer_relevance = value_layer_relevance.permute(0, 2, 1, 3).contiguous().view(new_shape)

        hidden_state_relevance1 = self.compute_linear_3(module.key, module_name + ".key", key_layer_relevance)
        hidden_state_relevance2 = self.compute_linear_3(module.value, module_name + ".value", value_layer_relevance)
        hidden_state_relevance3 = self.compute_linear_3(module.query, module_name + ".query", query_layer_relevance)

        return (hidden_state_relevance1 + hidden_state_relevance2 + hidden_state_relevance3) / 3

    def compute_attention(self, module, module_name, relevance):
        bert_output_hidden_relevance, bert_output_input_relevance = self.compute_bert_output(module.output,
                                                                                             module_name + ".output",
                                                                                             relevance)
        attention_relevance = self.compute_self_attention(module.self, module_name + ".self",
                                                          bert_output_hidden_relevance)

        return (attention_relevance + bert_output_input_relevance) / 2  # FIX IT

    def compute_bert_layer(self, module, module_name, relevance):
        bert_output_hidden_relevance, bert_output_input_relevance = self.compute_bert_output(module.output,
                                                                                             module_name + ".output",
                                                                                             relevance)
        intermediate_relevance = self.compute_linear_3(module.intermediate.dense, module_name + ".intermediate",
                                                       bert_output_hidden_relevance)
        attention_relevance = self.compute_attention(module.attention, module_name + ".attention", (
                intermediate_relevance + bert_output_input_relevance) / 2)  # FIX IT

        return attention_relevance

    def explain(self, inputs, target_outputs):
        self.register_hooks(self.model, self.layer_values_global)
        self.model = self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            _, logits = outputs[:2]
            logits_softmax = F.softmax(logits, dim=1)
            self.relevance_logits = torch.zeros(logits.shape, dtype=torch.float, device="cpu")
            for i, output_neuron in enumerate(target_outputs):
                self.relevance_logits[i][output_neuron[0]] = 0.5

            ner_outputs_relevance = self.compute_linear_3(self.model.classifier, "classifier",
                                                          self.relevance_logits)

            encoder_layers_next_relevance = [ner_outputs_relevance]
            attentions = []
            self_attentions = []
            for i, layer_module in reversed(list(enumerate(self.model.bert.encoder.layer))):
                encoder_layers_next_relevance.append(
                    self.compute_bert_layer(layer_module, "bert.encoder.layer." + str(i),
                                            encoder_layers_next_relevance[-1]))
                attentions.append(self.layer_values_global["bert.encoder.layer." + str(i) + ".attention"]["output"])
                self_attentions.append(
                    self.layer_values_global["bert.encoder.layer." + str(i) + ".attention.self.dropout"]["input"])

        return encoder_layers_next_relevance, attentions, self_attentions
