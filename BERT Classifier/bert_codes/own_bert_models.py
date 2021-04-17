from transformers.modeling_bert import *


# Function to select model based on parameters passed
def select_model(type_of_model,path,weights=None,label_list=None):
    if(type_of_model=='weighted'):
        model = SC_weighted_BERT.from_pretrained(
        path, # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification             # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
        weights=weights
    )
    elif(type_of_model=='normal'):
        model = BertForSequenceClassification.from_pretrained(
          path, # Use the 12-layer BERT model, with an uncased vocab.
          num_labels = 2, # The number of output labels--2 for binary classification             # You can increase this for multi-class tasks.   
          output_attentions = False, # Whether the model returns attentions weights.
          output_hidden_states = False, # Whether the model returns all hidden-states.
        )
    elif(type_of_model=='multitask'):
        model = BertForMultitask.from_pretrained(
          path, # Use the 12-layer BERT model, with an uncased vocab.
          num_labels = 2, # The number of output labels--2 for binary classification             # You can increase this for multi-class tasks.   
          output_attentions = False, # Whether the model returns attentions weights.
          output_hidden_states = False, # Whether the model returns all hidden-states.
          label_uniques=label_list
        )
    else:
        print("Error in model name!!!!")
    return model

        

# Class for weighted bert for sentence classification
class SC_weighted_BERT(BertPreTrainedModel):
    def __init__(self, config,weights):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.weights=weights
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss(weight=torch.tensor(self.weights).cuda())
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

# BERT for multitask learning
class BertForMultitask(BertPreTrainedModel):
    def __init__(self, config, label_uniques):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout_list=[]
        self.classifier_list=[]
        self.label_uniques=label_uniques
        for ele in self.label_uniques:
            self.dropout_list.append(nn.Dropout(config.hidden_dropout_prob))
            self.classifier_list.append(nn.Linear(config.hidden_size, ele))
        self.dropout_list=torch.nn.ModuleList(self.dropout_list)
        self.classifier_list=torch.nn.ModuleList(self.classifier_list)
        
        print("done")
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]
        logits_list=[]
        for i in range(len(self.label_uniques)):
            output_1 = self.dropout_list[i](pooled_output)
            logits = self.classifier_list[i](output_1)
            logits_list.append(logits)
        outputs = (logits_list,) + outputs[2:]  # add hidden states and attention if they are here
        loss=0
        for i in range(len(self.label_uniques)):
            # label=torch.nn.functional.one_hot(labels[:,i])
            label=labels[:,i]
            loss_fct = CrossEntropyLoss(reduction='mean').cuda()
            loss += loss_fct(logits_list[i].view(-1, self.label_uniques[i]), label.view(-1))
        outputs = (loss,) + outputs




        # if labels is not None:
        #     if self.num_labels == 1:
        #         #  We are doing regression
        #         loss_fct = MSELoss()
        #         loss = loss_fct(logits.view(-1), labels.view(-1))
        #     else:
        #         loss_fct = CrossEntropyLoss(weight=torch.tensor(self.weights).cuda())
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #     outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
