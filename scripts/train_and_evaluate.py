from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import DatasetDict, Dataset
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import nltk

# Step 1: Load and Preprocess Data
def load_translation_data():
    def read_file(en_file, pl_file):
        with open(en_file, 'r', encoding='utf-8') as f_en, open(pl_file, 'r', encoding='utf-8') as f_pl:
            en_sentences = f_en.readlines()
            pl_sentences = f_pl.readlines()
        assert len(en_sentences) == len(pl_sentences), "Mismatched dataset sizes!"
        return {"pl": pl_sentences, "en": en_sentences}

    train = read_file("data/splits_random/train.pl", "data/splits_random/train.en")
    val = read_file("data/splits_random/val.pl", "data/splits_random/val.en")
    test = read_file("data/splits_random/test.pl", "data/splits_random/test.en")

    return DatasetDict({
        "train": Dataset.from_dict(train),
        "validation": Dataset.from_dict(val),
        "test": Dataset.from_dict(test)
    })

data = load_translation_data()

# Step 2: Load Pre-trained Tokenizer and Model
model_name = "Helsinki-NLP/opus-mt-pl-en"
#model_name = "distilbert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 3: Tokenization Function
def preprocess_function(examples):
    # Tokenize the input sentences (Polish)
    # print("Examples['pl']:", examples["pl"])
    # print("Type of Examples['pl']:", type(examples["pl"]))
    # print("Examples['en']:", examples["en"])
    # print("Type of Examples['en']:", type(examples["en"]))
    
    inputs = list(examples["pl"])
    targets = list(examples["en"])

    # Tokenize inputs and targets
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")["input_ids"]

    model_inputs["labels"] = labels
    return model_inputs

# Tokenize the dataset
# tokenized_data = DatasetDict({
#         "train": preprocess_function(data['train']),
#         "validation": preprocess_function(data['validation']),
#         "test": preprocess_function(data['test'])
#     })

# Tokenize the dataset
tokenized_data = data.map(preprocess_function, batched=True, remove_columns=["pl", "en"])

#
print("Tokenized Dataset (Train) Example:")
print(tokenized_data["train"][0])  # Inspect the first example in the train dataset

#print("Keys in Dataset:", tokenized_data["train"].column_names)  # Check column names
print("Keys in Dataset Features:", tokenized_data["train"].features.keys())
#

# Step 4: Define Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./translation_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
    predict_with_generate=True,
    logging_dir="./logs"
)

# Step 5: Fine-Tune the Model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    tokenizer=tokenizer,
)

# Uncomment the following line to fine-tune the model (time-consuming step).
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_model")

# Step 6: BLEU Evaluation Function
def compute_bleu(references, predictions):
    """
    Calculates BLEU score using NLTK's BLEU implementation.

    Args:
        references: List of reference translations (list of lists of tokens).
        predictions: List of predicted translations (list of tokens).

    Returns:
        The BLEU score for the corpus.
    """
    references_tokenized = [[ref.split()] for ref in references]
    predictions_tokenized = [pred.split() for pred in predictions]
    return corpus_bleu(references_tokenized, predictions_tokenized)

# Step 7: Generate Translations for Test Set
def generate_translations(model, tokenizer, test_data):
    predictions = []
    references = []
    for example in test_data:
        input_text = example["translation"]["pl"]
        reference_text = example["translation"]["en"]
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs)
        prediction_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(prediction_text)
        references.append(reference_text)
    return references, predictions

# Load Pre-trained Model Without Fine-Tuning
pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load Fine-Tuned Model (If Fine-Tuning Is Complete)
# fine_tuned_model = AutoModelForSeq2SeqLM.from_pretrained("./fine_tuned_model")

# Generate Translations for Test Set Using Both Models
references, pretrained_predictions = generate_translations(pretrained_model, tokenizer, data["test"])
# _, fine_tuned_predictions = generate_translations(fine_tuned_model, tokenizer, data["test"])

# Step 8: Calculate BLEU Scores
pretrained_bleu = compute_bleu(references, pretrained_predictions)
# fine_tuned_bleu = compute_bleu(references, fine_tuned_predictions)

# Print Results
print("BLEU Score for Pre-trained Model:", pretrained_bleu)
# print("BLEU Score for Fine-Tuned Model:", fine_tuned_bleu)

# Step 9: Print Example Translations
print("\nExample Translations:")
for i in range(3):
    print(f"Polish: {data['test'][i]['translation']['pl']}")
    print(f"Reference (English): {references[i]}")
    print(f"Pre-trained Model Prediction: {pretrained_predictions[i]}")
    # print(f"Fine-Tuned Model Prediction: {fine_tuned_predictions[i]}")
    print()