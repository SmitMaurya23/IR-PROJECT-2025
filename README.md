---

# 📘 **README**

## 📁 **1. Download and Prepare the Dataset**

1. Download the `data` folder from the shared drive.
   Place it in the **root directory** of this project.

2. Download the `data_parsed` folder from the shared drive.
   Place it inside the **kg/** directory:

```
kg/data_parsed/
```

---

## 📦 **2. Install Dependencies**

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

Make sure your environment supports CUDA if you are using GPU-based models.

---

## 🗄️ **3. Neo4j Setup**

1. Download and install Neo4j Desktop or Neo4j Server from:
   [https://neo4j.com/download/](https://neo4j.com/download/)

2. Create a new Neo4j database and follow the official documentation to start the server.

3. Inside the **kg/** folder, open all Python files and replace:

```python
NEO4J_USER = "your_user"
NEO4J_PASSWORD = "your_password"
```

with your actual Neo4j credentials.

---

## 🧱 **4. Build the Knowledge Graph**

Navigate to:

```
kg/ingest_kg/
```

Run the ingestion scripts **in the following order**:

```bash
python ingest_parts.py
python ingest_chapters.py
python ingest_sections.py
python ingest_articles.py
```

Once all scripts finish successfully, your **Knowledge Graph is fully constructed**.

---

## 🔧 **5. Running the Base Hybrid Model**

### **5.1 Download Checkpoints**

Download all model checkpoints from the shared drive and place them inside:

```
checkpoints/
```

### **5.2 Set Environment Variables**

Set your Hugging Face and WandB tokens:

```bash
export HF_TOKEN="your_huggingface_token"
export WANDB_API_KEY="your_wandb_api_key"
```

**Note:** Ensure you have model access approvals for required models.

---

## 🚀 **6. Run the Base Pipeline**

Open and execute the Jupyter Notebook:

```
pipeline.ipynb
```

Run all cells sequentially.
This will generate and save:

* BGE embeddings
* Step 1 classifier
* RankLLaMA results
* Final ensemble models

Once complete, you are ready for inference.

---

## 🔍 **7. Inference Pipeline**

Run `infer.ipynb` until the section titled:

```
"Step 3 LLM Inference"
```

Before running Step 3, execute the LLM inference script:

```bash
sh run_step3.sh
```

Or run each command from:

```
llm_infer_commands.txt
```

After the script completes, return to `infer.ipynb` and execute the remaining cells.

Your final output will be stored here:

```
checkpoints/final/R05_submission.jsonl
```

---

## 🔄 **8. Select Dataset Split (R04 / R05 / R06)**

In both notebooks (`pipeline.ipynb` and `infer.ipynb`), update:

```python
SELECTED_ID = "R04"  # or R05, R06
```

Set it according to the dataset you want to run.

---

## 🧠 **9. Running the KG-Integrated Models**

To run the models with Knowledge Graph expansion:

1. Open and execute:

```
pipeline_with_kg.ipynb
```

2. Run:

```bash
sh run_step3.sh
```

3. Then execute:

```
infer_with_kg.ipynb
```

Follow the same order as the base pipeline.

---

## 🎉 **10. You're All Done!**

After running the full pipeline, the final predictions and logs will be generated under the `checkpoints/` directory.
You now have:

* Base Hybrid Model results
* KG-integrated model results
* Fully reproducible inference pipeline

For any issues, check Neo4j logs or confirm model checkpoint paths.

---


