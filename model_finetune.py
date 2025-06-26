from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def fine_tune_model():
    model_name = "roberta-base"
    dataset = load_dataset("glue", "sst2")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def preprocess(example):
        return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

    encoded = dataset.map(preprocess, batched=True)
    encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    args = TrainingArguments(
        output_dir="./model_output",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        num_train_epochs=1,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded["train"].shuffle(seed=42).select(range(1000)),
        eval_dataset=encoded["validation"].select(range(200)),
    )

    trainer.train()
    trainer.save_model("./model_output")
    tokenizer.save_pretrained("./model_output")
    print("✅ RoBERTa fine-tuning complete!")

if __name__ == "__main__":
    fine_tune_model()