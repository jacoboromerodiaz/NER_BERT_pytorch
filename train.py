from datasets import load_dataset
from itertools import chain

from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch
from torch.nn.utils.rnn import pad_sequence

import joblib

import config, model, dataset, engine

def process_data(dataset):
  enc_tag = preprocessing.LabelEncoder()
  enc_tag.fit(list(chain(*dataset["bio_tags"])))
  tag = [enc_tag.transform(sentence) for sentence in list(dataset["bio_tags"])]
  sentences = dataset["tokens"]
  return sentences, tag, enc_tag

def collate_fn(batch):
  token_ids_stack = [x['token_ids'] for x in batch]
  attention_mask_stack = [x['attention_mask'] for x in batch]
  token_type_ids_stack = [x['token_type_ids'] for x in batch]
  target_tags_stack = [x['target_tags'] for x in batch]

  token_ids_pad = pad_sequence(token_ids_stack, batch_first=True)
  attention_mask_pad = pad_sequence(attention_mask_stack, batch_first=True)
  token_type_ids_pad = pad_sequence(token_type_ids_stack, batch_first=True)
  target_tags_pad = pad_sequence(target_tags_stack, batch_first=True)

  return {
      "token_ids": token_ids_pad,
      "attention_mask": attention_mask_pad,
      "token_type_ids": token_type_ids_pad,
      "target_tags": target_tags_pad,
      }

if __name__ == "__main__":
  dataset_ner = load_dataset("DFKI-SLT/few-nerd", 'supervised')
  train_dataset = dataset_ner["train"]
  dataset_bio = dataset.bio_tagging(train_dataset)  
  
  train_sentences, train_tag, train_enc_tag = process_data(dataset_bio["train"])
  valid_sentences, valid_tag, valid_enc_tag = process_data(dataset_bio["validation"])
  test_sentences, test_tag, test_enc_tag = process_data(dataset_bio["test"])
  num_tag = len(list(train_enc_tag.classes_))
  
  meta_data = {
        "enc_tag": train_enc_tag,
    }
  
  joblib.dump(meta_data, "meta.bin")

  train_dataset = dataset.NERDataset(
      texts=train_sentences, tags=train_tag
  )
  train_data_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=0, collate_fn= collate_fn
  )
  valid_dataset = dataset.NERDataset(
      texts=valid_sentences, tags=valid_tag
  )
  valid_data_loader = torch.utils.data.DataLoader(
      valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=0
      , collate_fn= collate_fn,
  )

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.NERModel(num_tag=num_tag)
  model.to(device)

  param_optimizer = list(model.named_parameters())
  no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
  optimizer_parameters = [
      {
          "params": [
              p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
          ],
          "weight_decay": 0.001,
      },
      {
          "params": [
              p for n, p in param_optimizer if any(nd in n for nd in no_decay)
          ],
          "weight_decay": 0.0,
      },
  ]

  num_train_steps = int(len(train_sentences) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
  optimizer = AdamW(optimizer_parameters, lr=1e-4, no_deprecation_warning=True)
  scheduler = get_linear_schedule_with_warmup(
      optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
  )

  train_loss = []
  valid_loss = []
  for epoch in range(1,config.EPOCHS+1):
    print(f'Epoch {epoch} of {config.EPOCHS}')
    accuracy_epoch = []
    loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
    train_loss.append(loss)
    if epoch % 2 == 0 and epoch !=0:
      val_loss, targets, outputs = engine.eval_fn(valid_data_loader, model, device)
      metrics_dict = {class_i: {"precision":[], "recall": [], "f1score":[]} for class_i in train_enc_tag.classes_}
      metrics = precision_recall_fscore_support(targets, outputs,
                                                labels=list(range(num_tag)),
                                                average=None)
      for class_idx, class_i in enumerate(train_enc_tag.classes_):
          metrics_dict[class_i]["precision"] = metrics[0][class_idx]
          metrics_dict[class_i]["recall"] = metrics[1][class_idx]
          metrics_dict[class_i]["f1score"] = metrics[2][class_idx]

          print(f"{class_i} -> Precision: {metrics_dict[class_i]['precision']}, "
                f"Recall: {metrics_dict[class_i]['recall']}, F1score: {metrics_dict[class_i]['f1score']}")
      
      torch.save(model.state_dict(), f"{config.MODEL_PATH}_{epoch}")

    if epoch == config.EPOCHS:
      print(f"Train loss: {train_loss}")
