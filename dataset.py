import datasets
import torch

dataset = datasets.load_dataset("DFKI-SLT/few-nerd", 'supervised')
train_dataset = dataset["train"]
coarse_labels = train_dataset.features["ner_tags"].feature.names

def bio_tagging(dataset: datasets.arrow_dataset.Dataset):
  bio_tags_matrix = []
  for ner_tags in dataset["ner_tags"]:
    bio_tags = []
    intermediate = 0
    for ner_tag in ner_tags:
      if ner_tag != 0 and intermediate == 0:
        bio_tags.append(f"B-{coarse_labels[ner_tag]}")
        intermediate = 1
      elif ner_tag != 0 and intermediate == 1:
        if coarse_labels[ner_tag] != bio_tags[-1][2:]:
          bio_tags.append(f"B-{coarse_labels[ner_tag]}")
          continue
        bio_tags.append(f"I-{coarse_labels[ner_tag]}")
      else:
        intermediate = 0
        bio_tags.append("O")
    bio_tags_matrix.append(bio_tags)

  return {"bio_tags": bio_tags_matrix}

class NERDataset:
  def __init__(self,texts,tags):
    self.texts = texts
    self.tags = tags

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, item):
    text = self.texts[item]
    tags = self.tags[item]

    token_ids = []
    target_tags = []

    #Now we tokenize for bert
    for i, w in enumerate(text):
      inputs = TOKENIZER.encode_plus(
          w,
          add_special_tokens = False #CLS tokens avoided so we can create the targets
      )

      input_len = len(inputs["input_ids"])
      token_ids.extend(inputs["input_ids"])
      target_tags.extend([tags[i]]*input_len)

    token_ids = [101] + token_ids + [102]
    target_tags = [0] + target_tags + [0]

    attention_mask = [1] * len(token_ids)
    token_type_ids = [0] * len(token_ids)

    return {
        "token_ids": torch.as_tensor(token_ids, dtype= torch.long),
        "attention_mask": torch.as_tensor(attention_mask, dtype= torch.long),
        "token_type_ids": torch.as_tensor(token_type_ids, dtype= torch.long),
        "target_tags": torch.as_tensor(target_tags, dtype= torch.long),
    }
