from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains.qa_with_sources.stuff_prompt import template

# Define a custom prompt to guide the LLM's persona and behavior
# This ensures the model acts as a real estate expert and strictly follows the provided context
custom_qa_prompt_template = """You are a real estate and mortgage expert assistant. Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Instructions:
- Base your answer strictly on the provided context
- If the context doesn't contain the answer, say "I don't have enough information to answer that"
- Include specific numbers, dates, and rates when available
- Cite sources when mentioning statistics

Answer:"""

# Combine our custom behavior prompt with the default LangChain QA-with-sources template
new_template = custom_qa_prompt_template + template

# Create the final PromptTemplate object to be used by the generation chain
PROMPT = PromptTemplate(template=new_template, input_variables=["summaries", "question"])
