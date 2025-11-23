# ClauseEase: AI-Based Contract Language Simplifier
 
> **"Clear Contracts. Confident Decisions."**

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Backend-Flask-lightgrey)
![Firebase](https://img.shields.io/badge/Database-Firebase_Firestore-orange)
![AI Models](https://img.shields.io/badge/AI-HuggingFace_Transformers-yellow)

**ClauseEase** is an AI-driven solution designed to bridge the gap between complex legal documents and the general public. It automates the extraction, analysis, and simplification of legal clauses into plain English, fostering transparency and enabling informed decision-making for non-lawyers, business professionals, and legal researchers.

---

## 📑 Table of Contents
- Abstract
- Key Features
- System Architecture
- AI & NLP Models
- Tech Stack
- Installation & Setup
- Screenshots
- Future Enhancements
- Team
- References
  
---

## 📝 Abstract
Legal contracts are often written in dense, specialized jargon that creates barriers for individuals without legal expertise. ClauseEase addresses this by leveraging advanced NLP (Natural Language Processing) to:
1.  Ingest documents (PDF, DOCX, TXT).
2.  Detect and classify legal clauses.
3.  Recognize specific legal terminology.
4.  Rewrite complex text into understandable plain English while preserving legal intent.

---

## ✨ Key Features
* **📄 Multi-Format Ingestion:** Supports uploading of PDF, DOCX, and TXT files.
* **🔍 Automated Clause Detection:** Identifies and categorizes specific legal clauses using BERT/T5 architectures.
* **🧠 Legal Term Recognition:** Flags complex legal terminology and provides definitions/context using spaCy.
* **✍️ Language Simplification:** Translates intricate legal text into plain English using Transformer models.
* **📊 Report Generation:** Produces downloadable reports and visual analytics of the contract analysis.
* **🔐 User Authentication:** Secure login and data management via JWT and Firebase.
* **🎛️ Admin Dashboard:** A managed panel for reviewing analysis history and user activities.

---

## 🏗 System Architecture
ClauseEase employs a multi-layered architecture:
1.  **Frontend:** HTML/CSS/JavaScript interface for user interaction.
2.  **Backend:** Flask (Python) server acting as the central hub/API.
3.  **AI Engine:** Orchestrates calls to Hugging Face Transformers and NLP libraries.
4.  **Database:** Google Firebase Cloud Firestore (NoSQL) for real-time data synchronization and storage.

---

## 🤖 AI & NLP Models
The core intelligence of ClauseEase is built upon the following specific models and libraries:

| Module | Model / Technology | Purpose |
| :--- | :--- | :--- |
| **Clause Detection** | `google/flan-t5-small` | Classifies text into specific legal clause categories. |
| **Simplification** | `prajjwall/bert-mini` | Rewrites complex sentences into plain English. |
| **Term Recognition** | `en_core_web_sm` (spaCy) | Named Entity Recognition (NER) for legal terms. |

---

## 🛠 Tech Stack

### **Backend & AI**
* **Language:** Python
* **Framework:** Flask
* **ML Frameworks:** PyTorch, Transformers (Hugging Face), spaCy, NLTK
* **Libraries:** `numpy`, `scikit-learn`, `fitz` (PyMuPDF), `python-docx`, `matplotlib`

### **Frontend**
* **Core:** HTML5, CSS3, JavaScript
* **Interaction:** REST API calls via Fetch

### **Database & Cloud**
* **Database:** Google Firebase (Cloud Firestore)
* **Auth:** Json Web Token (JWT)

---

## 🚀 Installation & Setup

### Prerequisites
* Python 3.10 or higher
* `pip` (Python Package Manager)
* A Google Firebase project with Firestore enabled.

### Steps

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/tulu-g559/ClauseEase.git
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Ensure your `requirements.txt` includes:* `flask`, `firebase-admin`, `torch`, `transformers`, `nltk`, `spacy`, `pymupdf`, `python-docx`, `numpy`, `scikit-learn`....

4.  **Download NLP Models**
    ```bash
    # Download the spaCy model
    python -m spacy download en_core_web_sm
    ```

5.  **Configure Firebase**
    * Download your `serviceAccountKey.json` from Firebase Console.
    * Place it in the root directory or configure the path in `app.py`.

6.  **Run the Application**
    ```bash
    python app.py
    ```
    Access the app at `http://127.0.0.1:3000`.

---

## 📸 Screenshots

| **1. Home Page** | **2. Analysis Results** |
| :---: | :---: |
| ![Home Page](https://github.com/user-attachments/assets/075350ee-b798-4d7c-bfc6-216596b2a4f8) | ![Analysis Results](https://github.com/user-attachments/assets/3fcae6de-a1b0-47f0-b1d2-074cfe9086c1) |
| **3. See History** | **4. Admin Dashboard** |
| ![See History](https://github.com/user-attachments/assets/5e256022-f49e-4da7-983c-1fb52bd89f5f) | ![Admin Dashboard](https://github.com/user-attachments/assets/a5d70914-1048-4c31-af57-06625bdd0d6d) |



---

## 🔮 Future Enhancements
* **Multilingual Support:** Extending models to handle non-English contracts.
* **Real-time Comparison:** Comparing clauses across different document versions.
* **Legal Chatbot:** Integration of a conversational AI assistant for specific legal queries.
* **Docker Support:** Full containerization for easier deployment.

---

**Mentor:** Dr. A. Kalaivani

---

## 📚 References
1.  Devlin, J., et al. "BERT: Pre-training of deep bidirectional transformers." (2019).
2.  Google AI. "Text-to-Text Transfer Transformer (T5)." (2020).
3.  Explosion AI. "spaCy: Industrial-strength NLP."
4.  Hugging Face. "Transformers Documentation."
