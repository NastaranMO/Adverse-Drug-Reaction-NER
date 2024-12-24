from transformers import GPTNeoForCausalLM, GPT2Tokenizer

model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

def generate_text(prompt, max_new_tokens=100):
    # Tokenize input with attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Generate output with proper settings
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.7,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = f"""
You are tasked with extracting adverse drug rections in the text from the social media text froums like AskAPatisnt. 

Important Notes:
- Use entities as follows:
  - Drug
  - Disease
  - ADR
  - Symptom
  - Finding
- Rest of the things you find in the text that are not entities that I mentioned above you must tag as "O".
- Do the tagging in order and do not change the order.
- For each token in the text, you must provide a tag.

Examples:
1. Text: "I have severe headaches and muscle pain."
   Tags: "O O B-ADR I-ADR O B-ADR I-ADR"

2. Text: "I am taking Lipitor and feel dizzy."
   Tags: "O O O B-Drug O O B-ADR"

Now, annotate the following text:
Text: "I am taking Lipitor and feel"
Tags:
"""

# Generate output
output = generate_text(prompt)
print("Generated Output:", output)
