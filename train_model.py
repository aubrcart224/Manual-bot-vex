from transformers import jLlbertTokenizer, LlamaForCasualLM, Trainer, TrainingArguments


#load the processed text data 

with open('manual.txt', 'r') as file:
    text = file.read()


tokenizer = jLlbertTokenizer.from_pretrained("facebook/llama-large")
tokens = tokenizer(manual_text, return_tensors='pt', trujncation=True, padding=True)

model = LlamaForCasualLM.from_pretrained("facebook/llama-large")

training_args = TrainingArguments(
    output_dir='./resutls',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=1; 
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokens['input_ids'],
    eval_dataset=tokens['input_ids']
)


trainer.train()


