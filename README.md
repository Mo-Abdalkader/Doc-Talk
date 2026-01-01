# 🤖 Doc-Talk

### Chat with Your Documents Using AI - 100% Free!

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://doc-talk.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Doc-Talk** is a free, AI-powered RAG (Retrieval-Augmented Generation) system that lets you upload documents and have intelligent conversations about their content. Simply upload a PDF, DOCX, or TXT file, and start asking questions!

---

## ✨ Features

- 📄 **Multi-Format Support** - PDF, DOCX, and TXT files
- 🧠 **Smart AI Responses** - Powered by Google Gemini Pro
- 🔍 **Semantic Search** - FAISS vector database for accurate retrieval
- 💬 **Chat Interface** - Natural conversation flow
- 🔒 **Privacy First** - Documents processed in session only
- 💰 **100% Free** - No hidden costs or subscriptions
- ⚡ **Fast Processing** - Efficient chunking and embedding
- 🎨 **Clean UI** - Built with Streamlit

---

## 🚀 Live Demo

👉 **[Try Doc-Talk Now!](https://docy-talky.streamlit.app/)** 👈

---

## 🎯 How It Works

1. **Upload** your document (PDF, DOCX, or TXT)
2. **Process** - Doc-Talk splits and indexes your content
3. **Ask** questions about your document
4. **Get** AI-powered answers based on the content

```
Your Document → Text Extraction → Smart Chunking → Vector Embeddings 
→ FAISS Storage → Query → Similarity Search → Context Retrieval 
→ Gemini AI → Answer
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **LLM** | Google Gemini Pro |
| **Embeddings** | HuggingFace Sentence Transformers |
| **Vector Store** | FAISS |
| **Framework** | LangChain |
| **Parsing** | PyPDF2, python-docx |

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key ([Get it free](https://makersuite.google.com/app/apikey))

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/Mo-Abdalkader/Doc-Talk.git
cd doc-talk
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the app**
```bash
streamlit run app.py
```

4. **Enter your API key**
   - Open the app in your browser
   - Enter your Gemini API key in the sidebar
   - Start chatting with your documents!

---

## ☁️ Deploy Your Own

Deploy your own instance on Streamlit Cloud for free:

1. **Fork this repository**

2. **Get Gemini API Key**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a free API key

3. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Select your forked repository
   - Set main file as `app.py`

4. **Configure Secrets**
   - In Streamlit Cloud dashboard, go to **Settings → Secrets**
   - Add your API key:
   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```

5. **Deploy!** 🎉

---

## 📋 Usage

### Document Constraints
- **Max File Size**: 10 MB
- **Max Characters**: 100,000
- **Supported Formats**: PDF, DOCX, TXT

### Example Questions
- "What is the main topic of this document?"
- "Summarize the key points"
- "What does it say about [specific topic]?"
- "Find information about [keyword]"

---

## 🔐 Privacy & Security

- ✅ API keys stored securely in Streamlit secrets
- ✅ Documents processed in session only (not saved)
- ✅ No data persistence or logging
- ✅ Each user session is isolated
- ✅ No API keys exposed in code

---

## 📊 Project Structure

```
doc-talk/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignore rules
└── .streamlit/
    └── secrets.toml      # API keys (local only, not committed)
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🐛 Known Issues & Limitations

- First-time embedding download may take a moment
- Large documents (>50MB) may timeout
- Context limited to top 3 most relevant chunks
- Gemini API rate limits apply (generous free tier)

---

## 🔮 Roadmap

- [ ] Support for Excel and PowerPoint files
- [ ] Multi-document chat capability
- [ ] Chat history export (PDF/TXT)
- [ ] Highlighted answer sources
- [ ] Custom chunk size configuration
- [ ] Support for multiple LLM providers
- [ ] Document comparison feature

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) - Amazing framework for data apps
- [Google AI](https://ai.google.dev/) - Free Gemini API access
- [HuggingFace](https://huggingface.co/) - Open-source embeddings
- [LangChain](https://www.langchain.com/) - RAG infrastructure
- [FAISS](https://github.com/facebookresearch/faiss) - Efficient vector search

---

## 📧 Contact

Have questions or suggestions? Feel free to:
- Open an issue
- Submit a pull request
- Star ⭐ this repository if you find it helpful!

---

## 🌟 Star History

If you find Doc-Talk useful, please consider giving it a star! ⭐

---

**Built with ❤️ by the community | Powered by 100% free tools**

[🚀 Try the Demo](https://docy-talky.streamlit.app/) | [📖 Documentation](https://docy-talky.streamlit.app/) | [🐛 Report Bug](https://github.com/Mo-Abdalkader/Doc-Talk/issues)
