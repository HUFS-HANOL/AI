# AI
This repository collects the key components of the NLP module that plays a central role in our Capstone Design project.

## File Tree
```
📦capstone             #Root File
 ┣ 📂checkpoint        #Model Checkpoints
 ┣ 📂data              
 ┃ ┗ 📜poems.txt       #Preprocessed Dataset
 ┣ 📜.gitignore        #Checkpoints,Dataset
 ┣ 📜README.md         
 ┣ 📜config.py         #Configs
 ┣ 📜dataset.py        #Load Dataset
 ┣ 📜generate.py       #Generate Responses
 ┣ 📜model.py          #Model Architecture
 ┣ 📜preprocess.py     #Data Preprocessing
 ┗ 📜train.py          #Train Our Models
 ```

## Purpose
We are developing 'HAN-OL', a poetry recommendation service designed to deliver emotional comfort and healing

> 🌿 **Hanol** is a poetry recommendation service crafted to bring emotional comfort and healing.
>
> 🧠 Our goal is to generate poems that resonate with each user’s **unique psychological state** — not just text, but words that understand.
>
> 💬 To achieve this, we fine-tune `KoGPT2` with a focus on **emotional nuance and empathy**.
>
> 🔧 A *custom-designed loss function* helps guide the model to better reflect subtle human feelings in its poetic output.
>
> — Let machines write, but let them write **with feeling.**