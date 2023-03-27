# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %pip install -r requirements.txt --quiet

# +
import torch
from pathlib import Path
from PIL import Image

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
# -

# #### Load model and preprocessors

# +
# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)
# model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device=device, is_eval=True)
# -

# #### Preprocess image and text inputs

# +
# raw_image = Image.open("./menu_data/bad/babybeer.jpeg").convert("RGB")
# display(raw_image.resize((596, 437)))

statements = [
    "depicts a child or portrays objects, images, or cartoon figures that primarily appeal to persons below the legal purchase age",
    "uses the name of or depicts Santa Claus",
    'promotes alcohol use as a "rite of passage" to adulthood',
    "uses brand identification—including logos, trademarks, or names—on clothing, toys, games, game equipment, or other items intended for use primarily by persons below the legal purchase age",
    "portrays persons in a state of intoxication or in any way suggests that intoxication is socially acceptable conduct",
    "makes curative or therapeutic claims, except as permitted by law",
    "makes claims or representations that individuals can attain social, professional, educational, or athletic success or status due to beverage alcohol consumption",
    "degrades the image, form, or status of women, men, or of any ethnic group, minority, sexual orientation, religious affiliation, or other such group?",
    "uses lewd or indecent images or language",
    "employs religion or religious themes?",
    "relies upon sexual prowess or sexual success as a selling point for the brand",
    "uses graphic or gratuitous nudity, overt sexual activity, promiscuity, or sexually lewd or indecent images or language",
    "associates with anti-social or dangerous behavior",
    "depicts illegal activity of any kind?",
    'uses the term "spring break" or sponsors events or activities that use the term "spring break," unless those events or activities are located at a licensed retail establishmen',
]

txts = [text_processors["eval"](statement) for statement in statements]
    

# -

imgs = {
        img_path: vis_processors["eval"](Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        for img_path in Path('./discus/').glob('**/*.jpg')
}

# #### Compute image-text matching (ITM) score

model

# +
import pandas as pd

df = pd.DataFrame(columns=['img_parent','img_name','model','preprocess',*txts])
df.to_csv(Path('./vqa.csv'),mode='w')


for img_path, img in list(imgs.items()):
    thumb = Image.open(img_path)
    thumb.thumbnail((200,200))
    display(thumb)
    data={
        'img_parent': img_path.parent,
        'img_name': img_path.name,
        'model': 'blip2_image_text_matching',
        'preprocess': 'pretrain',
    }
    for txt in txts:
        itm_output = model({"image": img, "text_input": txt}, match_head="itm")
        itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
        print(f'The {img_path.name} and "{txt}" are matched with a probability of {itm_scores[:, 1].item():.3%}')
        data[txt]=itm_scores[:, 1].item()
    df = pd.concat([pd.DataFrame([data], columns=df.columns), df], ignore_index=True)
    
    df.to_csv(Path('./vqa.csv'),mode='w')

display(df)
# -

df.to_excel('vqa.xlsx')

for img_path in Path('./discus/').glob('**/*.[pdf|jpg]'):
    print(img_path)
    vis_processors["eval"](Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    for txt in txts:
        itc_score = model({"image": img, "text_input": txt}, match_head='itc')
        print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)


