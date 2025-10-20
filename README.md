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
> Ensure you have a Python version >=3.7.1 , <3.11 installed. Any other version will **NOT** work due to the implementation of CleanRL.
> Make sure this is the **ONLY** version of python installed.
> Version 3.10.2 is recommended: https://www.python.org/downloads/

Verify your version below: 

**Windows:**
```bash
python --version
```
**Linux/macOS:**
```bash
python3 --version
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

It should **NOT** look like this:

```bash
Sayeshen-Govender-Honours-Project-main/
└── Sayeshen-Govender-Honours-Project-main/
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
If this is the case, make sure to move the inner `Sayeshen-Govender-Honours-Project-main` folder directly into your downloads and use **this** as the root directory.

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

**Windows (Powershell/VS Code Terminal):**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
```bash
.venv\Scripts\Activate.ps1
```

**Windows (CMD/ Command Prompt):**
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
wandb disabled
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
> All of the models have already been fully trained and will be loaded when you download the file. You can immediately test them, so there is no need to retrain any.
> Retraining these models will take **several** hours.
> We first have instructions on what needs to be changed to  **test** our models, and thereafter are instructions on how to **retrain** our models.

In `main.py`:

<img width="1923" height="480" alt="image" src="https://github.com/user-attachments/assets/cb4f4338-6dd6-49ad-a191-43865040c0e5" />


You may change `env_id` to either "Hopper-v4", "HalfCheetah-v4" or "Walker2d-v4", depending on which environment you wish to evaluate.
You can also change `algo` to either "PPO" or "RPO"
You may then modify `gravity_scale` or `mass_scale` based on which perturbation you wish to apply. Only apply one at a time.
At this point you can already run `main.py` to evaluate any of our **Case One** experiments.

If you wish to test our **Case Two** models, then some code needs to be uncommented below:

<img width="1754" height="756" alt="image" src="https://github.com/user-attachments/assets/f33ffdd5-72a6-4172-9346-22ca6b7af6b1" />


Comment out, and uncomment based on which case you wish to evaluate, i.e., if you want to test **Case Two** with a gravity perturbation, comment out the lines with the "CASE 1" comments, and uncomment the "CASE 2 GRAV" comment.

Do the same below:

<img width="1749" height="194" alt="image" src="https://github.com/user-attachments/assets/c4407631-5142-4873-a142-ea0ff3f08c3a" />

<img width="1665" height="178" alt="image" src="https://github.com/user-attachments/assets/e616d1ff-0d43-4df9-8ede-88980d21489b" />

Now you should be able to perform testing on any of our pretrained models.

**If you wish to verify that training works as inteded, you may uncomment the following**:
<img width="1989" height="645" alt="image" src="https://github.com/user-attachments/assets/ff40f3e4-929e-4b08-9297-5710634883a2" />
<img width="1732" height="515" alt="image" src="https://github.com/user-attachments/assets/fe681a20-e46e-415a-a73d-b9b892a33379" />

Now for generating the training curves, we go to `curves.py`. 

**Windows:**
```bash
python curves.py
```
**Linux/macOS:**
```bash
python3 curves.py
```

Running `curves.py` without any changes will generate the training curves for **Case One** training, and save them in the `plots/` file.
If you wish to generate the curves for **Case Two**, you will need to uncomment the follwing:

<img width="1779" height="176" alt="image" src="https://github.com/user-attachments/assets/c818f7c9-5443-400e-b325-4b68298c4a62" />
<img width="1836" height="681" alt="image" src="https://github.com/user-attachments/assets/59b6e645-1471-4128-ae34-4050eb9dcd81" />
<img width="1641" height="251" alt="image" src="https://github.com/user-attachments/assets/e45ca02f-24b5-4c05-8f8b-97cd5793d70f" />
<img width="1778" height="246" alt="image" src="https://github.com/user-attachments/assets/59d061f6-e833-481b-9383-b3c9975e3f52" />
<img width="1404" height="244" alt="image" src="https://github.com/user-attachments/assets/5e3c57a2-a7a9-47a3-961f-c2fd54bca893" />

You can them run `curves.py` again for these cases.

**Windows:**
```bash
python curves.py
```
**Linux/macOS:**
```bash
python3 curves.py
```



