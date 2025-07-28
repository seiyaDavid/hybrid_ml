# Project Requirements Document (PRD)  
**Title:** Theme Classification and Chat System for CSV Issue Summaries

## Overview  
This system processes two or more CSV files containing issue summaries, extracts themes using embedding and clustering techniques, and allows users to interact with themes across datasets through a web UI and API.

---

## Requirements Table

| **Section**              | **Requirement**                                                                                                                                                           |
|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Input**                | Accept CSV files via API or UI upload with one column containing free-text issue summaries. Column name may vary.                                                         |
| **Preprocessing**        | - Extensive preprocessing: case normalization, whitespace cleanup, stopwords removal, special character stripping.<br>- Handle missing/null summaries robustly.           |
| **Embedding**            | Use **Amazon Titan Embeddings G1 - Text** to convert preprocessed summaries into dense vector representations.                                                             |
| **Theme Clustering**     | - Apply **UMAP** for dimensionality reduction.<br>- Use **HDBSCAN** to cluster similar summaries into potential themes.<br>- Leverage LangChain abstraction for modularity. |
| **Theme Description**    | Use **Claude-Sonnet via AWS Bedrock** to generate a name + description for each cluster/theme.                                                                             |
| **Zero-Shot Classification** | Use Claude-Sonnet via AWS Bedrock to classify each summary as “Data Quality Issue” or not using a zero-shot classification approach.                                     |
| **Compare Across Files** | - Detect overlapping and exclusive themes across two files.<br>- Enable querying shared and exclusive themes.<br>- Store results in a local vector database (FAISS).       |
| **Chat with Theme**      | - Enable user to click a theme and “chat” with it using a RAG pipeline powered by Claude-Sonnet and LangChain.<br>- Display linked summaries as part of the context.       |
| **Add Examples**         | Allow adding manual examples to themes for fine-tuning understanding.                                                                                                     |
| **APIs**                 | RESTful API (via **FastAPI**) to:<br>- Upload CSVs<br>- Trigger embedding, clustering, and classification<br>- Access themes<br>- Query/chat with themes                   |
| **UI**                   | - Build web UI using **Flask (Jinja2) + HTML + JavaScript + CSS**.<br>- Include upload interface, theme explorer, summary viewer, chat box.                               |
| **Project Structure**    | Modular ML structure with `/data`, `/services`, `/models`, `/routes`, `/utils`, `/configs`, `/templates`, and `/static`.                                                 |
| **Security**             | Store API credentials securely via `credentials.yaml`. Use role-based access and environment configs.                                                                      |
| **LangChain**            | Use LangChain to orchestrate embeddings, Claude calls, RAG-based chat, and chaining steps together.                                                                        |
| **Logging & Monitoring** | Use structured logging via Python’s `logging` library. Capture API activity, exceptions, model latency.                                                                   |
| **Scalability & Deployment** | - Containerize app using Docker.<br>- Use Gunicorn + Uvicorn for FastAPI deployment.<br>- Async processing for heavy tasks.                                                 |

---

## Future Enhancements
- Add multilingual support via translation embeddings.
- Incremental learning by storing new summaries and re-clustering periodically.
- Allow custom theme curation from the user.

---

## Tech Stack Summary

| Component          | Technology                         |
|--------------------|-------------------------------------|
| Embeddings         | Amazon Titan G1 - Text              |
| LLM                | Claude-Sonnet (AWS Bedrock)         |
| Vector DB          | FAISS                               |
| Clustering         | UMAP + HDBSCAN                      |
| App Backend        | FastAPI                             |
| UI Frontend        | Flask (Jinja2) + HTML + JS + CSS    |
| Orchestration      | LangChain                           |
| Security & Secrets | YAML (`credentials.yaml`)           |
| Logging            | Python `logging` module             |
| Deployment         | Docker, Gunicorn, Uvicorn           |

