---
title: CytoSight

---

<div align="center">
  <h1>🔬 CytoSight</h1>
  <p><strong>Automated Disease Diagnosis from Microscopic Cell Images</strong></p>
  
  [![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
  [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue?style=for-the-badge)](https://huggingface.co/)
  [![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)
</div>

---

## 🌟 Overview

**CytoSight** is an advanced, automated medical diagnostic platform designed to analyze microscopic cell images. By leveraging cutting-edge deep learning, CytoSight assists medical professionals by providing rapid, accurate, and explainable disease diagnoses from digital pathology scans.

## 🧠 Model Architecture & Training

At the core of CytoSight is our **custom AI model**, rigorously trained on a massive dataset of **90,000 images** spanning across:
- 🔬 **Histopathology**
- 🩸 **Hematology**

This extensive training allows the model to generalize well across different tissue types and cellular structures, ensuring high reliability in clinical scenarios.

## ⚙️ Three-Stage Diagnosis Process

Our system employs a robust **three-stage diagnosis process**:

1. **Region Identification**: The model first scans the microscopic image to identify and localize critical regions of interest.
2. **Status Determination**: Next, it evaluates the status of each identified region to determine whether the cells are normal or abnormal.
3. **Disease Staging**: Finally, if a region is classified as abnormal, the model performs a deeper analysis to determine the specific stage or severity of the disease.

## 🔍 Explainability Module

We believe in transparent AI. CytoSight's **Explainability Module** provides:
- **Visual Attention Maps**: Highlighting the exact cellular regions that contributed to the diagnosis.
- **Textual Reasoning**: Generating plain-text explanations describing *why* a specific diagnosis was made, helping bridge the gap between complex AI predictions and actionable medical insights.

## 💻 Web Application

The frontend is a beautifully crafted, responsive web application that allows users to:
- Securely **upload** microscopic cell images.
- Receive **real-time** diagnoses with rich visualizations.
- Review and track past analyses in their **Diagnosis History**.

## 🚀 Deployment & Tech Stack

CytoSight is built for scalability and performance:
- **Backend**: High-performance **FastAPI** server managing the inference pipelines.
- **Frontend**: Modern **React** / **Vite** architecture with seamless UI/UX.
- **Database**: Robust data management and secure authentication powered by **Supabase**.
- **Deployment**: Fully deployed as a Dockerized Space on **Hugging Face**, ensuring robust availability.
