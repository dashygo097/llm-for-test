import numpy as np

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset

model_name = "./gemma-2b-it-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# data with prefix
'''
# zero-shot
data = {
    "text": [
        "Classify the following sentence as Positive or Negative:\n\nSentence: This sentence is Positive.",
        "Classify the following sentence as Positive or Negative:\n\nSentence: This sentence is Negative.",

    ],
    "label": [
        "positive",
        "negative",
    ]
}
'''

'''
# 2-shot 
data = {
    "text": [
        "Classify the following sentence as Positive or Negative:\n\nSentence: 'I love this movie.'\nLabel:",
        "Classify the following sentence as Positive or Negative:\n\nSentence: 'I do like the main characters.'\nLabel:",
        "Classify the following sentence as Positive or Negative:\n\nSentence: 'I hate this movie.'\nLabel:",
        "Classify the following sentence as Positive or Negative:\n\nSentence: 'How can this movie be awful like this?'\nLabel:",

    ],
    "label": [
        "positive",
        "positive",
        "negative",
        "negative",
    ]
}
'''


# 4-shot
data = {
    "text": [
        "Classify the following sentence :\n\nSentence: 'I love this movie.'\nLabel:",
        "Classify the following sentence :\n\nSentence: 'I do like the main characters.'\nLabel:",
        "Classify the following sentence :\n\nSentence: 'I hate this movie.'\nLabel:",
        "Classify the following sentence :\n\nSentence: 'How can this movie be awful like this?'\nLabel:",
        "Classify the following sentence :\n\nSentence: 'One of the film's most striking aspects is its exploration of the human psyche and the consequences of one's actions.'\nLabel:",
        "Classify the following sentence :\n\nSentence: 'However, the film's dark tone and bleak outlook may not appeal to all audiences.'\nLabel:",
        "Classify the following sentence :\n\nSentence: 'Viewers who prefer more straightforward narratives and happier endings may find the film's dark tone and complex storyline difficult to digest.'\nLabel:",
        "Classify the following sentence :\n\nSentence: 'The movie is lush and beautiful , and the actors are well-chosen.'\nLabel:",
    ],
    "label": [
        "positive",
        "positive",
        "negative",
        "negative",
        "positive",
        "negative",
        "negative",
        "positive",
    ]
}


# prefix = [
#     "Classify the following sentence as Positive or Negative:\n\nSentence: ",
# ]
prefix = " "

dataset = Dataset.from_dict(data)

def data_process(examples):
    padding = 'max_length'
    max_length = 128

    inputs = [prefix + ex for ex in examples['text']]
    targets = [ex for ex in examples['label']]

    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs


dataset = dataset.map(
    data_process,
    batched=True,
    remove_columns=dataset.column_names,
)

adapter_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)

model_with_adapter = get_peft_model(model, adapter_config)

training_args = TrainingArguments(
    output_dir='./outputs',
    per_device_train_batch_size=4,
    learning_rate = 1e-5,
    num_train_epochs=2,
    logging_steps=200,
    remove_unused_columns=False
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

trainer = Trainer(
    model=model_with_adapter,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

from datasets import Dataset, load_dataset

dataset = load_dataset("./data/plain_text")
prefix = "Classify the following sentence as Positive or Negative: \n\nSentence: "
# test_prompt = [
#     "Classify the following sentence as Positive or Negative:\n\nSentence: 'I am excited about the new project.'\nLabel:",
# ]

testset = dataset["test"].shard(num_shards=125, index=0)  # means the first 200 samples
eval_dataset = []
for i in range(len(testset["text"])):
    eval_dataset.append(prefix + testset["text"][i])


def extract_label(output_text):
    if "positive" in output_text  :
        return 1
    elif "negative" in output_text :
        return 0
    else:
        return np.random.randint(0,2)

def eval_acc():
    count = 0
    for i in range(200):
        inputs = tokenizer(eval_dataset[i], return_tensors="pt",
                           padding='max_length', truncation=True, max_length=2000)
        outputs = model_with_adapter.generate(**inputs, max_length=2048)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generated_text)
        print(extract_label(generated_text))
        if (extract_label(generated_text) == testset["label"][i]):
            count += 1

    return count / 200


print(eval_acc())

# 0-shot --------accuracy 0.685
# 2-shot --------accuracy 0.74
# 4-shot --------accuracy 0.775
