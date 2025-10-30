# 🎧 Spotify Churn Analytics Playground

This project analyzes **Spotify user churn** — understanding why users stop using the service based on behavioral data.  
It includes data preprocessing, cleaning, and visualization steps inside a Jupyter Notebook.

---

## 📁 Project Structure

```
SPOTIFY-CHURN-ANALYTICS/
│
├── venv/                       # Virtual environment (excluded from Git)
├── preprocessing.ipynb         # Jupyter notebook for data analysis
├── spotify_churn_dataset.csv   # Dataset used for analysis
└── requirements.txt            # Python dependencies
```

---

## ⚙️ Setup Instructions

Follow these steps to set up and run the project on your local machine.

### 1️⃣ Clone this repository
```bash
git clone https://github.com/nzrfrz/spotify-churn-analytics-playground.git
cd spotify-churn-analytics-playground
```

### 2️⃣ Create a virtual environment
Create your virtual environment **inside the project folder**.

**Windows:**
```bash
python -m venv venv
```

**macOS/Linux:**
```bash
python3 -m venv venv
```

### 3️⃣ Activate the virtual environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

Once activated, you’ll see `(venv)` in your terminal prompt.

---

### 4️⃣ Install dependencies
Make sure you’re inside the project folder, then run:
```bash
pip install -r requirements.txt
```

This will install all libraries required to run the notebook.

---

## 💻 Choose How to Run the Project

You can run this project either using **VS Code** or **Jupyter Notebook** — choose whichever you prefer:

### Option A: 🧩 Open in VS Code

1. Open this project in **Visual Studio Code**.  
3. Make sure you have the [**Jupyter**](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter), [**Python**](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) and [**Jupyter Notebook Renderers**](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter-renderers) extensions installed in VS Code.  
4. Open the file `preprocessing.ipynb`.  
5. Select your Python environment (click the **kernel selector** at the top right, then choose your `venv`).  
6. Click **Run All** or execute cells one by one to explore the analysis.

### Option B: 📘 Open in Jupyter Notebook

1. In the terminal (with venv activated), run:
   ```bash
   jupyter notebook
   ```
2. When the Jupyter window opens in your browser, click on `preprocessing.ipynb` to open it.

---

## 📊 Dataset

The dataset file:
```
spotify_churn_dataset.csv
```

It contains user activity and churn-related data.  
Ensure it is located in the **same folder** as the notebook before running any code cells.

---

## 🧩 Requirements

Main dependencies used:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `jupyter`

Full list is available in `requirements.txt`.

---

## 🙌 Author

Developed by **[Nizar Farizi]**