# import streamlit as st
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# st.set_page_config(page_title='Armspellcheck')

# header = st.container()
# toutput = st.container()

# with header:
#     st.title("Welcome to ArmSpellCheck")
#     st.text("Here you can check your mistakes...")


# @st.cache
# def model_loader():
#     model = AutoModelForSeq2SeqLM.from_pretrained("Artyom/ArmSpellcheck_beta")
#     return model


# def tokenizer_loader():
#     tokenizer = AutoTokenizer.from_pretrained("Artyom/ArmSpellcheck_beta")
#     return tokenizer


# def output(model, tokenizer):
#     inpt = st.text_input(label="", value="Ձեր տեքստը")
#     inputs = tokenizer([inpt], padding="longest", return_tensors="pt").input_ids
#     res = model.generate(inputs)
#     print(res)
#     return res[0]


# m = model_loader()
# t = tokenizer_loader()
# with toutput:
#     st.write(t.decode(output(m, t)))
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

header = st.container()
toutput = st.container()

with header:
    st.title("Welcome to ArmSpellCheck")
    st.text("Here you can check your mistakes...")


@st.cache
def model_loader():
    model = AutoModelForSeq2SeqLM.from_pretrained("Artyom/ArmSpellcheck_beta")
    return model


def tokenizer_loader():
    tokenizer = AutoTokenizer.from_pretrained("Artyom/ArmSpellcheck_beta")
    return tokenizer


def output(textt, model, tokenizer):
    inputs = tokenizer([textt], padding="longest", return_tensors="pt").input_ids
    res = model.generate(inputs)
    return res[0]


m = model_loader()
t = tokenizer_loader()
inputt = st.text_input(label="", value="Ձեր տեքստը")
text = inputt.split(" ")
with toutput:
    for i in text:
        st.write(t.decode(output(i, m, t)))
