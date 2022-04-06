import streamlit as st
from transformers import T5ForConditionalGeneration, AutoTokenizer

header = st.container()
tinput = st.container()

with header:
    st.title("Welcome to ArmSpellCheck")
    st.text("Here you can check your mistakes...")


@st.cache
def model_loader():
    model = T5ForConditionalGeneration.from_pretrained('./final-model_3.0', )
    return model


def tokenizer_loader():
    tokenizer = AutoTokenizer.from_pretrained('./final-model_3.0')
    return tokenizer


def output(model, tokenizer):
    inpt = st.text_input(label="", value="Ձեր տեքստը")
    inputs = tokenizer([inpt], padding="longest", return_tensors="pt").input_ids
    res = model.generate(inputs)
    return res[0]


m = model_loader()
t = tokenizer_loader()

st.write(t.decode(output(m, t)))
