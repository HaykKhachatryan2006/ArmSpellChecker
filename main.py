import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title='Armspellcheck')

header = st.container()
toutput = st.container()

with header:
    st.title("Welcome to ArmSpellCheck")
    st.text("Here you can check your mistakes...")


@st.cache(allow_output_mutation=True)
def model_loader():
    model = AutoModelForSeq2SeqLM.from_pretrained("Artyom/ArmSpellcheck_beta")
    return model


def tokenizer_loader():
    tokenizer = AutoTokenizer.from_pretrained("Artyom/ArmSpellcheck_beta")
    return tokenizer


def output(model, tokenizer):
    x = 256
    inputs = tokenizer([inpt], padding="longest", return_tensors="pt", max_length=x).input_ids
    res = model.generate(inputs, max_length=x)
    print(res)
    return res[0]


m = model_loader()
t = tokenizer_loader()
with toutput:
    inpt = st.text_input(label="", value="Ձեր տեքստը")
    st.write(t.decode(output(m, t))[5:-4])
