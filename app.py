import streamlit as st
import torch
from pathlib import Path
from PIL import Image
import pandas as pd
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor

# Load model and preprocessors
@st.cache(allow_output_mutation=True)
def load_model():
    # Setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)
    return model, vis_processors, text_processors

# Preprocess image and text inputs
def preprocess_inputs(model, vis_processors, text_processors):
    statements = [...]  # The same statements list as in the original code
    txts = [text_processors["eval"](statement) for statement in statements]
    imgs = {
        img_path: vis_processors["eval"](Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        for img_path in Path('./discus/').glob('**/*.jpg')
    }
    return txts, imgs

# Compute image-text matching (ITM) score
def compute_itm_scores(model, txts, imgs):
    df = pd.DataFrame(columns=['img_parent','img_name','model','preprocess',*txts])
    for img_path, img in imgs.items():
        data = {...}  # The same data dictionary as in the original code
        for txt in txts:
            itm_output = model({"image": img, "text_input": txt}, match_head="itm")
            itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
            data[txt] = itm_scores[:, 1].item()
        df = pd.concat([pd.DataFrame([data], columns=df.columns), df], ignore_index=True)
    return df

# Streamlit app
def app():
    st.title("Image-Text Matching with Streamlit")
    
    # Load model
    model, vis_processors, text_processors = load_model()
    
    # Preprocess inputs
    txts, imgs = preprocess_inputs(model, vis_processors, text_processors)
    
    # Compute ITM scores
    df = compute_itm_scores(model, txts, imgs)
    
    # Display results
    st.dataframe(df)
    
    # Save results to Excel
    if st.button("Save to Excel"):
        df.to_excel('vqa.xlsx')
        st.success("Results saved to vqa.xlsx")

if __name__ == "__main__":
    app()
