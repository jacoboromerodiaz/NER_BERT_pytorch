import torch
import torch.nn as nn
from config import MODEL

def loss_fn(output, target, mask, num_labels):
  lfn = nn.CrossEntropyLoss()
  active_loss = mask.view(-1) == 1
  active_logits = output.view(-1, num_labels)
  active_labels = torch.where(
      active_loss,
      target.view(-1),
      torch.tensor(lfn.ignore_index).type_as(target)
  )
  loss = lfn(active_logits, active_labels)
  return loss

class NERModel(nn.Module):
  def __init__(self, num_tag):
      super(NERModel, self).__init__()
      self.num_tag = num_tag
      self.bert = MODEL
      self.bert_drop = nn.Dropout(0.3)
      self.out_tag = nn.Linear(768, self.num_tag)

  def forward(self, token_ids, attention_mask, token_type_ids, target_tags):
      o1, _ = self.bert(token_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
      bo_tag = self.bert_drop(o1)
      output = self.out_tag(bo_tag)
      loss = loss_fn(output, target_tags, attention_mask, self.num_tag)

      return output, loss