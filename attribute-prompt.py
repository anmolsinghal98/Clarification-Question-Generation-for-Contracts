import torch
from transformers import pipeline
import pandas as pd

#This code used Dolly for obtaining CQGEN inferences

generate_text = pipeline(model="databricks/dolly-v2-12b", torch_dtype=torch.bfloat16,
                         trust_remote_code=True, device=0, return_full_text=True, max_new_tokens=500)

from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

# template for an instrution with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

# template for an instruction with input
prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}")

hf_pipeline = HuggingFacePipeline(pipeline=generate_text)

llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)
llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)

#Attribute prompt with additional instructions about input and output
instruction = """
You are a legal assistant tasked with analyzing contractual sentences for ambiguities.
Task: Given a contractual sentence, identify all vague phrases present in the sentence. Also identify all other details and references missing from the context of the sentence.
Generate a list of clarification questions that should be asked to clarify each vague phrase and missing detail identified.
Output format should be as follows-
Vague Phrases: <Comma-seperated List>
Missing Details: <Comma-seperated List>
List of Clarification Questions: <Numbered List>
Given Sentence:
"""
"""
import time
start = time.time()

df = pd.read_csv('/workspace/data/Anmol/Clarification-Prompt/EMNLP-CUAD.csv')

c = 0
outputs = []
for i in range(len(df)):
    c += 1
    print(c)
    context = df.iloc[i]['Sentences'] + "```"
    context += "\n###Output: "
    output = llm_context_chain.predict(instruction=instruction, context=context).lstrip()
    outputs.append(output)


dic = {'Sentences': df['Sentences'], 'Dolly Output': outputs}
new_df = pd.DataFrame(dic)

new_df.to_csv('/workspace/data/Anmol/Clarification-Prompt/Dolly-Output-Attributes.csv', index = False)

end = time.time()
print("time",end - start)
