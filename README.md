# Legal AI Assistant - Assignment 1

This project is a Retrieval-Augmented Generation (RAG) system built for the LAWS90286 Legal AI course. The application allows users to upload a legal document and ask questions about its content, receiving structured, AI-powered analysis in return.

---

## Deployed Application URL

You can access the live application here:

**[请在这里粘贴你最终部署成功的 Streamlit Cloud 网址]**

---

## How to Use the Application

1.  **Provide API Credentials**: Upon loading the app, please use the sidebar on the left to enter your own OpenAI API key as required by the assignment brief.
2.  **Upload a Document**: Use the "+" icon on the left of the search bar to upload a legal document (PDF, DOCX, or TXT). Wait for the "Document indexed successfully!" message.
3.  **Ask a Question**: Type your question about the document into the central search bar.
4.  **Analyze**: Click the "➤" paper plane icon to receive a structured analysis based on the document's content.
5.  **Review History**: Your past questions and answers are stored in the sidebar for easy review.

---

## Test Case

This test case uses the Telstra "Our Customer Terms - Small Business General Terms" document to demonstrate the application's capabilities.

* **Document for Testing**:
    * **File Name**: `small-business-general-22122024.pdf`
    * **Download Link**: [Telstra Our Customer Terms](https://www.telstra.com.au/content/dam/tcom/our-customer-terms/business-government/pdf/small-business-general-22122024.pdf)

* **Test Steps**:
    1.  Upload the `small-business-general-22122024.pdf` document.
    2.  Ask the following questions one by one and observe the results.

* **Test Questions and Expected Outcomes**:

    1.  **Question**: `Summarise the customer's obligations regarding the use of services as outlined in the 'Using our services' section.`
        * **Expected Outcome**: The application should provide a summary of key obligations, including using the service for its intended purpose, taking responsibility for all use, and not using it for illegal acts, citing relevant clauses.

    2.  **Question**: `If Telstra's equipment at my premises is damaged, who is responsible for the cost of loss or damage?`
        * **Expected Outcome**: The application should correctly identify that the customer is responsible for the cost, citing Clause 3.11.

---

## Acknowledgements & Code Citation

This application was developed with the assistance of Google's Gemini large language model for UI design, debugging, and code generation.

The core RAG architecture is conceptually based on the principles demonstrated in the LlamaIndex starter tutorials, as recommended in the assignment brief.