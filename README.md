

Readme legal ai assistant · MD
Legal AI Assistant
面向长篇法律文件的检索增强问答工具:上传合同或条款文件,针对具体问题定位相关条款并输出结构化分析。

A retrieval-augmented question answering tool for long legal documents. Upload a contract, ask a specific question, and get a structured analysis grounded in the retrieved clauses.

在线体验 / Live app: https://legal-ai-assignment-1-vudcjwxygnshdknkpvnwvh.streamlit.app/
Show Image
<img width="1416" height="643" alt="image" src="https://github.com/user-attachments/assets/a8fcc7a3-f809-4851-baf6-6af9ba0153f7" />
<img width="1395" height="596" alt="image" src="https://github.com/user-attachments/assets/f3c16b97-8610-47c1-88cd-f2ada7b28c87" />


应用采用「用户自备 API Key」模式,在侧边栏选择模型供应商并填入自己的 Key 即可使用。 The app runs on a bring-your-own-key model: pick a provider in the sidebar and paste your own API key.

这个工具解决什么问题 / What it does
一份一百多页的服务条款,想知道「客户在使用服务方面承担哪些义务」,人工翻阅成本很高。本工具将文档切分并建立向量索引,针对提问检索最相关的若干片段,再由模型生成固定格式的分析。

Reading a 100-page terms-of-service document to answer one specific question is slow. This tool indexes the document, retrieves the passages most relevant to a question, and asks the model to answer strictly from those passages.

回答固定为四段结构 / Every answer follows a fixed four-part structure:

Analysis	拆解问题的构成要素 / Break the question into its components
Relevant clauses	引用条款编号与原文 / Cite clause numbers and original wording
Direct answer	一句话结论 / One-sentence conclusion
Reasoning	条款如何支撑结论 / How the clauses support the conclusion
回答下方可展开「Show Cited Sources」,查看实际被检索到的原文片段及相似度分数——每个结论都可追溯到具体页码。

An expandable "Show Cited Sources" panel shows the passages actually retrieved and their similarity scores, so every conclusion is traceable to a page.

使用方法 / How to use
选择供应商并填入 API Key — 侧边栏顶部选择智谱 GLM / Google Gemini / OpenAI,填入对应的 Key / Pick a provider in the sidebar and paste the matching API key
上传文档 — 点击搜索框左侧的 「+」 图标,支持 PDF、DOCX、TXT;等待 "Document indexed successfully" 提示 / Click the "+" icon to the left of the search bar and wait for the indexing confirmation
输入问题 — 在中间的搜索框内提问 / Type your question in the search bar
获取分析 — 点击 「➤」 纸飞机图标 / Click the "➤" icon
查看来源 — 展开回答下方的 "Show Cited Sources",查看被检索到的原文与页码 / Expand "Show Cited Sources" to see the retrieved passages and page numbers
历史记录 — past questions and answers 保存在侧边栏 / Past questions and answers are kept in the sidebar
测试用例 / Test case
以澳大利亚电信运营商 Telstra 的《小微企业通用条款》为样本(100+ 页),两个问题分别检验归纳能力与事实定位能力:

Using Telstra's Small Business General Terms (100+ pages) as the sample document. The two questions test summarisation and factual retrieval respectively.

测试文档 / Test document: small-business-general-22122024.pdf

问题 / Question	预期结果 / Expected
Summarise the customer's obligations regarding the use of services as outlined in the 'Using our services' section.	归纳出按预期用途使用、对所有使用行为负责、不得用于违法目的等义务,并引用对应条款 / Summarises the key obligations with clause citations
If Telstra's equipment at my premises is damaged, who is responsible for the cost of loss or damage?	明确指出由客户承担,引用 Clause 3.11 / Identifies the customer as responsible, citing Clause 3.11
第二个问题的答案唯一且可核对,用于验证检索是否真正定位到了正确条款,而非泛泛作答。

The second question has a single verifiable answer, which is what makes it useful: it shows whether retrieval actually located the right clause rather than producing a plausible summary.

技术栈 / Built with
Python · Streamlit — 界面与部署 / UI and deployment
LlamaIndex — 文档切分、向量索引与检索 / chunking, indexing, retrieval
pdfplumber — PDF 逐页文本提取 / page-by-page PDF text extraction
多供应商支持 / Multi-provider — 智谱 GLM、Google Gemini、OpenAI,通过 OpenAI 兼容接口统一接入 / Zhipu GLM, Google Gemini and OpenAI, unified through their OpenAI-compatible endpoints
几个设计决定 / Design notes
为什么用 OpenAILike 而不是 OpenAI LlamaIndex 的 OpenAI 类会将模型名与内置白名单比对,任何第三方模型都会触发 ValueError: Unknown model。OpenAILike 不做该校验,因此接入任意 OpenAI 兼容接口只需修改 endpoint 和模型名。

The OpenAI class validates model names against a hard-coded list, so any third-party model raises ValueError: Unknown model. OpenAILike skips that check, which reduces switching providers to a change of endpoint and model name.

为什么自行提取 PDF 文本 SimpleDirectoryReader 依赖 pypdf,对部分 PDF 提取失败且会将大量 XMP 元数据(pdf:Keywords 等)附加到文档上。元数据默认参与向量化,当正文提取失败时,索引中唯一可读的内容就是元数据,导致检索结果全部指向元数据而非条款。改为使用 pdfplumber 逐页提取并剥离元数据后解决。

pypdf returns nothing for some PDFs and attaches a large block of XMP metadata to every document. That metadata gets embedded alongside the text, so when extraction fails it becomes the only readable content in the index and every answer quotes it. Extracting page by page with pdfplumber and stripping metadata fixes both problems.

已知边界 / Known limitation 检索式问答适合「在长文档中定位并归纳特定条款」,不适合「通读全文给出整体评价」——模型只能看到检索出的若干片段,从未读过完整文档。需要确定性判定的场景(如合同风险审查)应使用规则引擎而非本方案,参见 Contract Risk Auditor。

Retrieval-based QA is good at locating and summarising specific clauses in a long document. It is not suited to whole-document evaluation, because the model only ever sees the retrieved passages. Tasks that need deterministic, reproducible judgements belong in a rule engine instead — see the companion project above.

本地运行 / Running locally
bash
git clone https://github.com/chenxi-wang183/legal-ai-assignment-1.git
cd legal-ai-assignment-1
pip install -r requirements.txt
streamlit run app.py
开发说明 / Development note
作者为法学背景,无编程专业训练,项目通过大模型辅助编程(vibe coding)完成从架构设计、代码实现到线上部署的全过程,并独立完成模型接口迁移、依赖冲突与文档解析等问题的排查。

Written by a law student with no formal programming background, using LLM-assisted development throughout — architecture, implementation and deployment — including independently diagnosing the provider migration, dependency and document-parsing issues described above.

原型为墨尔本大学《法律与人工智能》(LAWS90286)课程作业,后续持续迭代。RAG 架构参考 LlamaIndex 官方入门教程;界面设计与代码实现借助 Google Gemini 完成。

Originally built for the Law and AI course (LAWS90286) at the University of Melbourne, maintained since. The RAG architecture follows the LlamaIndex starter tutorials; the interface and implementation were produced with the help of Google Gemini.


