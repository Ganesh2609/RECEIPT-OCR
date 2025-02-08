# **Optical Character Recognition of Unstructured Receipts**  

## **1. Introduction**  
This project presents an end-to-end pipeline for Optical Character Recognition (OCR) and structured data extraction from unstructured receipts. The pipeline enhances document quality through image denoising, performs text recognition, extracts key information, and converts extracted data into SQL queries for structured storage and retrieval. The approach integrates multiple deep learning models and natural language processing techniques to improve accuracy and efficiency in receipt processing.  

## **2. Objectives**  
- To develop a pipeline capable of extracting textual information from receipt images, including handwritten and printed text.  
- To implement a denoising mechanism that improves OCR performance by reducing image noise.  
- To utilize advanced key information extraction techniques to convert unstructured text into structured data.  
- To integrate a language model for translating English queries into SQL for efficient data retrieval.  

## **3. Methodology**  

### **3.1 Image Denoising Using Restormer**  
Restormer, an attention-based deep learning model, is employed for document denoising. It utilizes depthwise convolutions and attention mechanisms to enhance image quality, thereby improving OCR accuracy.  

### **3.2 Optical Character Recognition (OCR) Using DocTR**  
OCR is performed using the DocTR framework, which employs:  
- **DBNet** for detecting text regions within receipt images.  
- **CRNN (Convolutional Recurrent Neural Network)** for text recognition, ensuring high accuracy in extracting text from various formats and fonts.  

### **3.3 Key Information Extraction Using SDMGR**  
The Spatial Dual Modality Graph Reasoning (SDMGR) model is implemented to extract structured information from OCR outputs. It combines textual and spatial features using graph reasoning to improve the accuracy of key-value pair extraction.  

### **3.4 SQL Query Generation Using a Large Language Model (LLM)**  
The extracted structured data is converted into SQL queries using Gemini Pro, a transformer-based large language model. It translates natural language queries into SQL statements, enabling seamless data retrieval.  

## **4. System Architecture**  
The system follows a modular architecture consisting of:  
1. **Preprocessing Layer** – Denoising using Restormer.  
2. **OCR Layer** – Text detection and recognition using DocTR.  
3. **Information Extraction Layer** – Key-value extraction using SDMGR.  
4. **Query Processing Layer** – SQL generation using Gemini Pro.  
5. **Frontend Application** – User interface for uploading receipts and querying extracted data.  

## **5. Performance Evaluation**  
- **OCR Accuracy**: 76.97% (Using DocTR with DBNet and CRNN).  
- **Key Information Extraction F1 Score**: 93% (Using SDMGR).  
- **Denoising Performance**: Mean Squared Error (MSE) reduced to 0.0116.  

## **6. Applications**  
- Automated expense tracking and financial record-keeping.  
- Receipt digitization and structured storage for businesses.  
- Integration with financial management software for data analytics.  

## **7. Conclusion**  
The proposed pipeline successfully enhances OCR performance through denoising, accurately extracts key information from receipts, and enables efficient data retrieval through SQL query generation. The modular design ensures adaptability for real-world receipt processing applications.  

## **8. References**  
1. Zamir, S. W. et al. (2022). *Restormer: Efficient Transformer for High-Resolution Image Restoration.* CVPR.  
2. Liao, M. et al. (2020). *Real-Time Scene Text Detection with Differentiable Binarization.* AAAI Conference on Artificial Intelligence.  
3. Sun, H. et al. (2021). *Spatial Dual Modality Graph Reasoning for Key Information Extraction.* arXiv.  
4. Radford, A. (2018). *Improving Language Understanding by Generative Pre-Training.* OpenAI.  

## **9. Contributors**  
- **Arun Prasad T D**  
- **Ganesh Sundhar S**  
- **Hari Krishnan N**  
- **Shruthikaa V**  
