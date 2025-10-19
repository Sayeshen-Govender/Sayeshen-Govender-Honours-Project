# Evaluating-Generalization-in-Deep-Reinforcement-Learning-across-Continuous-Control-Tasks
A repository consisting of the code for the Honours Project, "Evaluating Generalization in Deep Reinforcement Learning across Continuous Control Tasks", by Sayeshen Govender

## Installation and Setup

Follow the steps below to set up and run this project locally.

---

### Step 1: Download the Repository

1. Click the **green “Code”** button at the top right of this page.  
2. Select **“Local” → “Download ZIP”**.  
3. Extract the ZIP file to your "**Downloads**.  

>  **Note:**
> Ensure you have Python 3.8+ installed.

**Windows:**
```bash
python -m pip install --upgrade pip
```
**Linux/macOS:**
```bash
python3 -m pip install --upgrade pip
```
> On **Windows**, unzipping may create a subfolder with the same name.  
> Make sure the root folder is named `Sayeshen-Govender-Honours-Project-main` and does **not** contain another subfolder with the same name.

The directory should look like this after unzipping:
```bash
Sayeshen-Govender-Honours-Project-main/
├── logs/
├── curves.py
├── main.py
├── ppo_continuous_action.py
├── rpo_continuous_action.py
├── test_ppo.py
├── test_rpo.py
├── train.py
├── utils.py
├── requirements.txt
└── README.md
```
---

### Step 2: Open in Code Editor

- Open your preferred **code editor** (e.g., VS Code).  
- Set the **working directory** to the extracted project folder (`Sayeshen-Govender-Honours-Project-main`).

**Windows:**
```bash
cd C:\Users\<YourName>\Downloads\Sayeshen-Govender-Honours-Project-main
```
Replace <YourName> with the name of your PC

**Linux/macOS:**
```bash
cd ~/Downloads/Sayeshen-Govender-Honours-Project-main
```

---

### Step 3: Create a Virtual Environment

**Windows:**
```bash
python -m venv .venv
```
**Linux/macOS:**
```bash
python3 -m venv .venv
```
---

### Step 4: Activate the Virtual Environment

**Windows:**
```bash
.venv\Scripts\activate
```
**Linux/macOS:**
```bash
source .venv/bin/activate
```
---

### Step 5: Install Dependencies

**Windows:**
```bash
pip install -r requirements.txt
```
**Linux/macOS:**
```bash
pip install -r requirements.txt
```
---

### Step 6: Disable Weights & Biases (wandb)
**Windows:**
```bash
set WANDB_MODE=disabled
```
**Linux/macOS:**
```bash
export WANDB_MODE=disabled
```
---

### Step 7: Run the Project

**Windows:**
```bash
python main.py
```
**Linux/macOS:**
```bash
python3 main.py
```


---

## Below are specific instructions for training and evaluating specific experiments, and generating training curves. If you do not wish to do this, then you may ignore the instructions below.

>  **Note:**
> All of the models have already been fully trained, and you can immediately test them.
> Retraining these models will take **several** hours.
> We first have instructions on what needs to be changed to  **test** our models, and thereafter are instructions on how to **retrain** our models.


