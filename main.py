import streamlit as st
# from transformers import T5ForConditionalGeneration, AutoTokenizer

header = st.container()
tinput = st.container()

with header:
    st.title("Welcome to ArmSpellCheck")
    st.text("Here you can check your mistakes...")

# model = T5ForConditionalGeneration.from_pretrained('C:/Users/Lenovo/Desktop/ArmSpellcheck/final-model',).cuda()
# tokenizer = AutoTokenizer.from_pretrained('C:/Users/Lenovo/Desktop/ArmSpellcheck/final-model')
# inputs = tokenizer(['այծակ'], padding="longest", return_tensors="pt").input_ids.cuda()
# res = model.generate(inputs)
# print(res[0])

with tinput:
    inpt = st.text_input("Enter your text...", )
    # inputs = tokenizer([inpt], padding="longest", return_tensors="pt").input_ids.cuda()
    # res = model.generate(inputs)
    # st.write(tokenizer.decode(res[0]))
    st.write(inpt)
