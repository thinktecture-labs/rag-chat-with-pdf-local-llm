# Chat with your PDF documents - optionally with a local LLM

## Installation
`pip install -r requirements.txt`

`pip install -U pydantic==1.10.9`

## Run it
`streamlit run chat.py`

## Running a local LLM
Easiest way to run a local LLM is to use LM Studio:
https://lmstudio.ai/

The LLM I use in my conference talks (works fine on a MBP M1 Max with 64GB RAM):
- TheBloke/dolphin-2.2.1-mistral-7B-GGUF/dolphin-2.2.1-mistral-7b.Q8_0.gguf
  - https://huggingface.co/TheBloke/dolphin-2.2.1-mistral-7B-GGUF
