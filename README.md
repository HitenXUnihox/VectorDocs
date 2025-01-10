# Document Processing & QA System

This is a web-based application for processing PDF documents and performing question-and-answer (Q&A) tasks based on the uploaded documents. The system utilizes Hugging Face embeddings and stores vectorized representations in a Pinecone vector database. It is built using Streamlit for the frontend interface.

---

## Features

- Upload PDF files to extract text.
- Chunk large documents into manageable pieces.
- Generate embeddings for document chunks using Hugging Face models.
- Store vector embeddings in Pinecone for efficient retrieval.
- Ask questions about the uploaded documents and get relevant answers.

---

## Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) for the UI.
- **Embedding Model**: [Hugging Face Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- **Vector Database**: [Pinecone](https://www.pinecone.io/).
- **Backend Libraries**:
  - [PyPDF2](https://pypi.org/project/PyPDF2/) for PDF text extraction.
  - [Sentence Transformers](https://www.sbert.net/) for generating vector embeddings.

---

# Installation
---
## Prerequisites

Ensure you have the following installed:

- Python 3.7 or higher
- Pip
---
## Steps

### 1. Clone the repository:

   ```bash
   git clone urilink
   cd document-qa-system
   ```
### 2. Install dependencies:

```bash
pip install -r requirements.txt
```
### 3. Set up your Pinecone API:

- Sign up at Pinecone and get your API key.
- Replace the api_key and region variables in the code with your Pinecone credentials.
### 4. Run the application:

```bash
streamlit run app.py
```
### 5. Open the app in your browser:

Streamlit will generate a local URL (usually http://localhost:8501).
---
## Usage
** Upload a PDF file using the "Upload a PDF Document" section.
** The document text will be extracted, chunked, and stored in the Pinecone database.
** Enter your question in the text input field, and the system will retrieve and display relevant answers.
---
## File Structure
```bash
document-qa-system/
│
├── app.py                # Main application code
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
```
---
## Dependencies
- Install the following Python libraries via requirements.txt:
---
## Customization
- Chunk Size: Adjust the chunk_size parameter in the chunk_text function to control the size of text chunks.
- Embedding Model: Replace the Hugging Face model (sentence-transformers/all-MiniLM-L6-v2) with another model if needed.
- Styling: Modify the st.markdown section for custom UI styling.
---
## Screenshots
![Output Image](file:///C:/Users/patel/Pictures/Screenshots/Screenshot%202025-01-10%20104308.png)
---
## License
- This project is licensed under the MIT License.
--- 
## Acknowledgments
- [Hugging Face](https://huggingface.co/) for the embedding models.
- [Pinecone](https://www.pinecone.io/) for vector database services.
- [Streamlit](https://streamlit.io/) for the interactive UI.
---
## Contributing
Feel free to fork this repository and submit pull requests for enhancements or bug fixes. Contributions are welcome!
---
## Contact
For questions or feedback, contact [hiten.patel@unihox.com](hiten.patel@unihox.com).