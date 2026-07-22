# AI Customer Support Chatbot - Final Project Report

## 1. Executive Summary
This document outlines the architecture, development, and deployment of the AI-powered Customer Support Chatbot. The system utilizes Retrieval-Augmented Generation (RAG) to provide fast, accurate, and context-aware responses to customer inquiries by pulling directly from processed historical support tickets and manuals.

## 2. System Architecture
The application is built on a modern, decoupled architecture:
- **Data Pipeline:** Raw data (historical tickets, FAQs) is ingested, cleaned, and vectorized.
- **AI Backend (RAG):** Built with LangChain and Hugging Face, integrating vector search (e.g., ChromaDB/Azure Cognitive Search) to retrieve context before generating a response.
- **API Layer:** A robust, asynchronous FastAPI application that securely serves the RAG model to any frontend interface.
- **Cloud Infrastructure:** Hosted on Microsoft Azure with complete CI/CD automation via GitHub Actions.

## 3. Security Implementation
Security is handled at the API layer. All requests to the core `/chat` endpoint must include a secure `X-API-Key` header. This ensures that only authorized frontend clients or internal systems can query the RAG model, preventing abuse and managing load.

## 4. Evaluation & Metrics
The model's outputs are rigorously tested using standard NLP metrics, including BLEU and ROUGE, tracked via MLflow to ensure responses maintain high relevance and factual accuracy without hallucination.
