import os
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

# Read data from csv_output file
data_list = []
for file in os.listdir("./text_annotations"):
    data = pd.read_csv(f"./text_annotations/{file}")
    # save data and its iob to a list
    example = {
        "Text": data["Sentence"][0],
        "IOB": data["IOB"][0],
        "File": file
    }
    data_list.append(example)

# The prompt for the model to extract ADRs from the text
prompt = f"""
You are tasked with extracting adverse drug rections in the text from the social media text froums like AskAPatisnt. 

Here are some examples for reference:

1. Text: "lethargy and leg soreness.,"
   ADRs: lethargy; leg soreness

2. Text: "aches and pains, muscular soreness in arms and shoulders."
   ADRs: "aches and pains; muscular soreness in arms and shoulders"

3. Text: "Stomach pains, gass, dry mouth and drowsiness. Wonder if arm rest wouldn't have delivered same result without the need to take drug."
   ADRs: "Stomach pains; dry mouth; drowsiness; gass"

4. Text: "sometimes I get cramps in thigh. Have taken the above since 1996 -- Can I begin to have side effects now, after all this time?."
   ADRs: "cramps in thigh"


Now, annotate the following text:
Text: "leg cramps at 40mg. went away at 20mg. ldl stayed lower. just reduce your dosage to retain benifits and stop side effects."
ADRs: 
"""



# Call the OpenAI API
response = client.chat.completions.create(model="gpt-4",
messages=[
    {"role": "system", "content": "You are an assistant trained to annotate text with IOB tags."},
    {"role": "user", "content": prompt}
],
max_tokens=700,
temperature=0.7)

# Extract and print the response
generated_tags = response.choices[0].message.content.strip()
# print("text", data_list[6]["Text"])
print("Generated Tags:", generated_tags)



# Sample output for reference
# Now, annotate the following text:
# ner ade,"after I had been on it for 3 months,started breaking out in blisters,and my face looked like it was on fire. the dr took me off of it. but it helped my cholestrol.",blisters
# Text: "after I had been on it for 3 months,started breaking out in blisters,and my face looked like it was on fire. the dr took me off of it. but it helped my cholestrol."
# ADRs:  "breaking out in blisters; face looked like it was on fire"

# Now, annotate the following text:
# Text: "Taking 40mg daily for several years. Cholesterol much lower,HDL higher,LDL below 100 and lower triglycerides."
# ADRs:  none correct

# Now, annotate the following text:
# Text: "leg cramps at 40mg. went away at 20mg. ldl stayed lower. just reduce your dosage to retain benifits and stop side effects."
# ADRs:  leg cramps, correct

