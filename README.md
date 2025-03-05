以下是一个适合你的 **`README.md`**，涵盖了项目介绍、安装步骤、数据管理说明、主要代码结构等内容。

---

## **Applied Data Science Coursework**

This repository contains coursework for applied data science, including data processing, statistical analysis, and machine learning models.

### **Project Structure**

```plaintext
📂 analyst/            # Main project directory
│── 📂 src/            # Source code
│   │── 📂 definition/  # Core definitions (constants, enums, etc.)
│   │── 📂 log/        # Logging configurations (excluded from Git)
│   │── 📂 main/       # Main scripts
│   │── 📂 parser/     # Codebook parsing and processing
│   │── 📂 pipeline/   # Data pipelines
│   │── 📂 util/       # Utility functions
│── poetry.lock        # Poetry dependency lock file
│── pyproject.toml     # Project dependencies and configurations
│── requirements.txt   # Exported dependencies for pip users
│── README.md          # Project documentation (this file)
📂 data/           # Data folder (ignored in Git, uploaded via WhatsApp)
📂 prefect/        # Prefect workflow and cache (ignored in Git)
```

---

## **Installation**

### **1. Clone this repository**

```sh
git clone git@github.com:jcfangc/applied-data-science-courcework.git
cd applied-data-science-courcework
```

### **2. Install dependencies**

If using Poetry:

```sh
poetry install
```

If using pip:

```sh
pip install -r requirements.txt
```

---

## **Data Management**

-   **The `data/` folder is ignored in Git** to prevent large file uploads.
-   The dataset will be shared **privately via WhatsApp**.
-   Once received, place the dataset inside the `data/` folder.

---

## **Usage**

### **Run the main script**

```sh
poetry run python src/main/main.py
```

or if using pip:

```sh
python src/main/main.py
```

### **Zip all JSON files**

```sh
poetry run python src/parser/zip_up.py
```

### **Run Prefect Workflow**

```sh
poetry run prefect deployment build src/pipeline/flow/compute_divergence_flow.py
```

---

## **Contribution**

1. **Create a branch**:
    ```sh
    git checkout -b feature-branch-name
    ```
2. **Make changes and commit**:
    ```sh
    git add .
    git commit -m "Describe your changes"
    ```
3. **Push to GitHub and create a pull request**:
    ```sh
    git push origin feature-branch-name
    ```

---

## **License**

This project is for educational purposes and is not licensed for commercial use.

---
