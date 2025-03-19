

# 1: Install Required Libraries
!pip install transformers datasets accelerate peft bitsandbytes

#  2: Load the Pre-Trained Model and Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Setting the model name here, other options can be given
model_name = "gpt2"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
print(f"Model on GPU: {next(model.parameters()).is_cuda}")

# Set the padding token
tok.pad_token = tok.eos_token

#  3: Using PEFT method LoRA
from peft import LoraConfig, get_peft_model

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn"],  # Target attention layers in GPT-2
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)

# 4: Prepare the Dataset
import pandas as pd
from datasets import Dataset

# Load the dataset
data_path = "/kaggle/input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCity.csv"
df = pd.read_csv(data_path)
df = df.dropna()
df = df[['dt', 'AverageTemperature', 'City', 'Country']]
df['dt'] = pd.to_datetime(df['dt'])

# Filter dataset to relevant countries and time period
countries_list = ['Germany', 'France', 'United Kingdom', 'United States', 'Canada', 'Australia', 'India', 'China', 'Japan', 'Brazil']
df = df[df['Country'].isin(countries_list)]
df = df[(df['dt'].dt.year >= 1900) & (df['dt'].dt.year <= 2000)]
print(f"Filtered dataset size: {len(df)} rows")

# Calculate temperature trends
def calc_trend(city_data):
    city_data = city_data.sort_values('dt')
    start_yr = city_data['dt'].min().year
    end_yr = city_data['dt'].max().year
    if len(city_data) >= 2:
        temp_start = city_data.iloc[0]['AverageTemperature']
        temp_end = city_data.iloc[-1]['AverageTemperature']
        temp_trend = temp_end - temp_start
        return pd.Series({'start_year': start_yr, 'end_year': end_yr, 'trend': temp_trend})
    return pd.Series({'start_year': None, 'end_year': None, 'trend': None})

trend_data = df.groupby(['City', 'Country']).apply(calc_trend, include_groups=False).reset_index()
trend_data = trend_data.dropna()
print(f"Trend dataset size: {len(trend_data)} rows")

# Check Berlin's trend
berlin_trend = trend_data[(trend_data['City'] == 'Delhi') & (trend_data['Country'] == 'India')]['trend'].values[0]
print(f"Berlin's actual trend: {berlin_trend:.2f}°C")

# Create instruction-response pairs
instr, resp = [], []
for _, row in trend_data.iterrows():
    q = f"What is the temperature trend in {row['City']}, {row['Country']} from {int(row['start_year'])} to {int(row['end_year'])}?"
    a = f"The temperature trend in {row['City']}, {row['Country']} from {int(row['start_year'])} to {int(row['end_year'])} was {row['trend']:.2f}°C."
    instr.append(q)
    resp.append(a)

qa_data = pd.DataFrame({'instruction': instr, 'response': resp})
dataset = Dataset.from_pandas(qa_data)
print("Dataset created")

# Combine instructions and responses
def format_qa(examples):
    return {'text': f"### Instruction: {examples['instruction']}\n### Response: {examples['response']}"}

dataset = dataset.map(format_qa, num_proc=4, desc="Formatting Q&A data")
print("Text combined")

# Tokenize the dataset
def tokenize(batch):
    return tok(batch['text'], padding='max_length', truncation=True, max_length=128)

tokenized_data = dataset.map(tokenize, batched=True, num_proc=4, desc="Tokenizing data")
print("Dataset tokenized")

# Split the dataset into train and eval sets
data_split = tokenized_data.train_test_split(test_size=0.1)
train_data, eval_data = data_split['train'], data_split['test']
print(f"Train size: {len(train_data)}, Eval size: {len(eval_data)}")

# 5: Fine-Tune the Model with Custom Training Loop
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from tqdm import tqdm
import time

train_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    num_train_epochs=5,
    fp16=True,
    learning_rate=1e-5,
    lr_scheduler_type='cosine',
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_data,
    data_collator=data_collator,
)

# Custom training loop
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=train_args.learning_rate)

steps_per_epoch = len(trainer.get_train_dataloader())
total_epochs = train_args.num_train_epochs
total_steps = steps_per_epoch * total_epochs

global_step = 0
for epoch in range(total_epochs):
    print(f"\nEpoch {epoch + 1}/{total_epochs}")
    progress_bar = tqdm(range(steps_per_epoch), desc="Training", unit="step")
    
    for step, batch in enumerate(trainer.get_train_dataloader()):
        batch = {k: v.to('cuda') for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        
        if global_step % 50 == 0:
            print(f"Step {global_step}/{total_steps}: Loss = {loss.item():.4f}")
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        progress_bar.update(1)
        global_step += 1
    
    progress_bar.close()

print("Training complete")

#  6: Evaluate the Model
query = "### Instruction: What is the temperature trend in Delhi, India from 1900 to 2000?\n### Response:"
inputs = tok(query, return_tensors='pt').to('cuda')
outputs = model.generate(**inputs, max_length=40, temperature=0.7, top_k=50, do_sample=True, no_repeat_ngram_size=2)
print(tok.decode(outputs[0], skip_special_tokens=True))

# Step 7: Save the Fine-Tuned Model
model.save_pretrained('./fine-tuned-gpt2-climate')
tok.save_pretrained('./fine-tuned-gpt2-climate')
