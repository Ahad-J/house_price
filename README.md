# 🏠 Pakistan House Price Prediction System

### *End-to-End Reproducible ML Workflow with DVC, Git, and Flask Deployment*

---

## 📘 Overview

This project implements a **complete machine learning workflow** for predicting house prices in Pakistan, using the **Pakistan House Price Dataset** from Kaggle.
It demonstrates how to build a **reproducible, version-controlled ML pipeline** with **DVC (Data Version Control)** and **GitHub**, and then deploy the trained model as a **Flask web application**.

---

## 🚀 Features

✅ End-to-end **data → model → deployment** workflow
✅ **DVC-based pipeline** for reproducible ML experiments
✅ **Parameter-driven training** using `params.yaml`
✅ **Model and data versioning** with DVC
✅ **Flask web app** for interactive prediction and REST API access
✅ **Local or remote DVC storage** for large files
✅ Easy to **retrain, reproduce, and redeploy**

---

## 🧩 Project Structure

```
house_price_project/
│
├── data/                     # Raw and versioned data (DVC tracked)
│   └── Entities.csv
│
├── models/                   # Trained models and encoders (DVC outputs)
│   ├── house_price_model.pkl
│   ├── label_encoders.pkl
│   ├── model_features.pkl
│   └── feature_field_map.pkl
│
├── src/                      # Source scripts
│   └── train.py              # Training pipeline
│
├── templates/                # HTML templates for Flask
│   ├── index.html
│   └── result.html
│
├── app.py                    # Flask app for web + API prediction
├── params.yaml               # Parameter configuration for reproducibility
├── dvc.yaml                  # DVC pipeline definition
├── dvc.lock                  # DVC pipeline lock (auto-generated)
├── metrics.json              # Model evaluation metrics (MSE, R²)
├── requirements.txt          # Python dependencies
├── .gitignore
├── .dvcignore
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/house_price.git
cd house_price
```

### 2️⃣ Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # on macOS/Linux
venv\Scripts\activate      # on Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 💾 DVC Setup

### 4️⃣ Initialize DVC (already done)

```bash
dvc init
```

### 5️⃣ Pull Data and Models (if not present)

```bash
dvc pull
```

This restores your dataset and trained model files from your configured DVC remote (local or cloud).

---

## 🧠 Model Training Pipeline

The training process is parameterized via `params.yaml` and automated using `dvc.yaml`.

### 6️⃣ Run Training with DVC

```bash
dvc repro
```

This command:

* Reads parameters from `params.yaml`
* Loads data from `data/Entities.csv`
* Trains the RandomForest model
* Evaluates metrics (MSE, R²)
* Saves model artifacts in `models/`
* Stores metrics in `metrics.json`

### 7️⃣ View Metrics

```bash
cat metrics.json
```

Example output:

```json
{
  "mse": 265211908247625.16,
  "r2": 0.8184
}
```

---

## 🌐 Flask Web Application

Once the model is trained and artifacts are saved, you can launch the web interface.

### 8️⃣ Run the Flask Server

```bash
python app.py
```

Server starts at:

```
http://127.0.0.1:5000
```

### 🖥️ Web UI

* Enter property details in the input form.
* Click **Predict** to view the predicted house price.

### 🧾 API Endpoint (Optional)

You can also make predictions via JSON API:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{
           "area": 3000,
           "property_type": "House",
           "city": "Lahore",
           "province_name": "Punjab",
           "purpose": "For Sale",
           "baths": 3,
           "bedrooms": 4,
           "latitude": 31.5,
           "longitude": 74.3
         }'
```

Response:

```json
{"prediction": 24500000.0}
```

---

## 🧾 params.yaml Example

```yaml
random_state: 42
test_size: 0.2
model:
  type: RandomForestRegressor
  n_estimators: 100
```

---

## 📊 Metrics Tracking

DVC automatically tracks model performance metrics in `metrics.json`.
You can compare different experiments using:

```bash
dvc metrics diff
```

---

## ☁️ DVC Remote Storage

To enable team collaboration and remote artifact tracking, configure a DVC remote:

### Example (Local Remote)

```bash
dvc remote add -d local_remote /home/user/dvc_storage
git add .dvc/config
git commit -m "Add local DVC remote"
dvc push
```

---

## 🧩 Git & Collaboration Workflow

1. Each team member clones the repository
2. Creates their own feature branch (e.g., `feature-api`)
3. Works on a module
4. Commits and pushes changes
5. Merges into `main` with conflict resolution if needed

---

## 🧠 Tech Stack

* **Python 3.10+**
* **Flask** – Web application
* **Scikit-learn** – Model training
* **Pandas / NumPy** – Data processing
* **DVC** – Data & model versioning
* **Git / GitHub** – Code version control
* **Joblib** – Model persistence

---

## 🏁 Results

| Metric  | Value       |
| ------- | ----------- |
| **MSE** | 2.65 × 10¹⁴ |
| **R²**  | 0.818       |

The trained model achieves an **R² score of 0.82**, showing strong predictive performance on test data.

---

## 💡 Future Enhancements

* Add **CI/CD** for automatic retraining on data updates
* Use **Docker** for containerized deployment
* Integrate **DVC experiments** for hyperparameter tuning
* Deploy to **Render / AWS / Heroku**

---

## 👨‍💻 Author
🔗 [GitHub: Ahad-J](https://github.com/Ahad-J)
