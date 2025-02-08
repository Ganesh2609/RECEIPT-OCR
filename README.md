### 📌 **Optical Character Recognition of Unstructured Receipts**  

![Project Banner](#) *(Add an image or GIF showcasing your project in action!)*  

### 🔍 **Overview**  
This project develops an **end-to-end pipeline for Optical Character Recognition (OCR)** with a **frontend application** for enhanced usability. The pipeline follows these key steps:  
- **📄 Image Denoising**: Uses **Restormer** for noise removal.  
- **🔍 OCR Processing**: Utilizes **DocTR** (DBNet for text detection & CRNN for text recognition).  
- **📊 Key Information Extraction (KIE)**: Employs **SDMGR (Spatial Dual Modality Graph Reasoning)**.  
- **🧠 SQL Query Generation**: Uses **Gemini Pro (LLM)** for natural language to SQL conversion.  

This system efficiently converts **unstructured receipt images** into **structured, actionable insights** for improved data retrieval and analysis.  

---

### 🏆 **Features**  
✔️ **Restormer**: Enhances image clarity for improved OCR accuracy.  
✔️ **DocTR OCR**: Uses DBNet and CRNN for high-accuracy text extraction.  
✔️ **SDMGR**: Graph-based reasoning for extracting structured information.  
✔️ **LLM (Gemini Pro)**: Converts extracted data into **SQL queries** for easy retrieval.  
✔️ **Frontend UI**: Interactive interface for users to upload receipts & view results.  

---

### 📊 **Results & Performance**  
📌 **OCR Accuracy**: 76.97% (DocTR OCR)  
📌 **Key Information Extraction F1 Score**: 93% (SDMGR)  
📌 **Denoising Performance (MSE)**: 0.0116 (Restormer)  

---

### 📜 **References**  
- [Restormer: Efficient Transformer for High-Resolution Image Restoration](https://arxiv.org/abs/2110.04621)  
- [DocTR: Deep Learning-Based OCR for Documents](https://github.com/mindee/doctr)  
- [DBNet for Scene Text Detection](https://arxiv.org/abs/2005.02357)  
- [SDMGR for Key Information Extraction](https://arxiv.org/abs/2103.14470)  
- [Gemini Pro LLM](https://ai.google.dev/)  

---

### 📌 **Contributors**   
👤 Ganesh Sundhar S  
👤 Arun Prasad T D 
👤 Hari Krishnan N  
👤 Shruthikaa V  

---

Would you like to include any additional details, such as API endpoints or frontend instructions? 🚀
