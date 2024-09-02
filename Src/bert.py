import torch

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import EvalPrediction
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

model_path = "./bert-base-uncased"
data_path = "./data/plain_text"


def BERT_LoadHandler(model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, problem_type="single_label_classification"
    )

    return tokenizer, model


dataset = load_dataset(data_path)

bert_tokenizer, bert_model = BERT_LoadHandler()


def data_process(example, tokenizer=bert_tokenizer):
    text = example["text"]
    label = example["label"]
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    encoding["label"] = torch.tensor(label, dtype=torch.float16).reshape(-1, 1).long()

    return encoding


encoded_dataset = dataset.map(
    data_process, batched=True, remove_columns=dataset["train"].column_names
)
encoded_testset = dataset.map(
    data_process, batched=True, remove_columns=dataset["test"].column_names
)
encoded_dataset.set_format("torch")
encoded_testset.set_format("torch")

print(encoded_dataset["train"])
print(encoded_testset.keys())



exit()
batch_size = 32
metric_name = "f1"

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

args = TrainingArguments(
    f"./data/output_format_bert",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

trainer = Trainer(
    bert_model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_testset["test"],
    tokenizer=bert_tokenizer,
    compute_metrics=compute_metrics
)

for name, param in bert_model.named_parameters():
    if not param.is_contiguous():
        print(f"Parameter {name} is not contiguous. Making it contiguous.")
        param.data = param.data.contiguous()
    else:
        print(f"Parameter {name} is already contiguous.")


torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer.train()
