import streamlit as st
from streamlit_option_menu import option_menu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(
    page_title="Armspellcheck",
)


@st.experimental_memo(show_spinner = True)
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


nav = option_menu(
    menu_title=None,
    options=["Home", "Project", "Contacts"],
    icons=["house", "calculator-fill", "envelope"],
    default_index=0,
    orientation="horizontal"
)

if nav == "Home":
    header = st.container()
    with header:
        st.title("About or program")
        st.markdown("Welcome user. We created a program for checking your mistakes. You can try out it in project.")
elif nav == "Project":
    header = st.container()
    toutput = st.container()
    ml = model_loader()
    m = ml
    t = tokenizer_loader()
    st.balloons()

    with header:
        st.title("Here you can check your mistakes...")

    with toutput:
        inpt = st.text_input(label="", value="Ձեր տեքստը")
        st.write(t.decode(output(m, t))[5:-4])
elif nav == "Contacts":
    st.snow()
    cheader = st.container()
    with cheader:
        st.title("Our contacts")
        st.info(
            "Hayk:"
        )
        st.info(
            "Artyom:"
        )
        st.info(
            "Hakob:"
        )
