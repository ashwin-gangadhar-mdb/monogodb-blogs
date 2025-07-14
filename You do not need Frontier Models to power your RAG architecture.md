**Blog Title:** You do not need Frontier Models to power your RAG architecture  
**Authors:** [Ashwin Gangadhar](mailto:ashwin.gangadhar@mongodb.com)  
**Reviewers: [ayaan@fireworks.ai](mailto:ayaan@fireworks.ai)**  
**Keywords: Firework, fine-tuning, RAG, Small language models(SLM)**

# 

# You do not need Frontier Models to power your RAG architecture

Frontier AI models are driving the widespread adoption of Generative AI by demonstrating unprecedented capabilities. However, their deployment often entails significant costs. The strategic partnership between MongoDB and Fireworks.AI addresses these cost implications by offering solutions that optimize performance and resource utilization. This collaboration leverages MongoDB's efficient data management alongside Fireworks.AI's model optimization tools to enhance speed and efficiency while minimizing operational expenses.

In the current AI environment, achieving high performance is crucial, but equally important is optimizing the Total Cost of Ownership (TCO). Businesses must focus on the price/performance ratio, ensuring that improvements in speed and efficiency lead to real cost savings.

This article will address the following topics:

* How to build an Agentic RAG using [Fireworks.ai](http://Fireworks.ai) hosted LLM and MongoDB Atlas for retrieval.  
* Strategies for optimizing Retrieval Augmented Generation (RAG) applications using MongoDB Atlas and Large Language Models (LLMs) through effective query and response caching.  
* Techniques using the Fireworks.AI platform focus on fine-tuning models, accelerating LLM inference, and reducing hardware needs.  
* Steps to finetune a pretrained SLM with PEFT techniques using Fireworks Platform.

Readers will gain a practical, in-depth strategy to improve AI performance while lowering costs. This will be demonstrated with examples and performance data.

## **Unlocking Efficiency and Performance with MongoDB and Fireworks.AI**

MongoDB Atlas is renowned for its flexible schema, efficient indexing, and distributed architecture, allowing organizations to scale their data infrastructure on demand. MongoDB Atlas is a general purpose database that focuses on highlighting flexibility, AI suitability, and ACID transactions. Users have the flexibility to run their application anywhere but making sure that there are no compromises made in the security aspects of working with it. MongoDB offers a comprehensive, secure, and efficient database solution for modern applications, catering to various technical and strategic needs.

Fireworks AI is recognized for its suite of technologies focused on optimizing the performance and efficiency of large language models (LLMs). Their offerings span model optimization tools, a specialized FireOptimizer framework, and innovative attention mechanisms like FireAttention. These solutions aim to enhance inference speeds, reduce operational costs, and improve resource utilization. Furthermore, Fireworks AI provides parameter-efficient fine-tuning methods and adaptive speculative execution to tailor models for specific applications. Their advancements also include optimized processing for long-context tasks and techniques to maximize throughput and cost-effectiveness in model serving. Fireworks also provides model serving functionality for select models that are readily available, also they do provide platform to host and serve custom implementations of LLM models for customers.

**Core Capabilities: FireOptimizer and FireAttention**

The **FireOptimizer** is Fireworks.ai’s adaptation engine for customizing AI model performance in production environments. It automates latency and quality optimization for unique inference workloads. It tailors performance across hardware, model, and software layers using techniques like customizable quantization, fine-tuning, and adaptive caching. Its hallmark feature, adaptive speculative execution, automatically trains workload-specific draft models to parallelize token generation, achieving up to 3x latency improvements compared to generic speculative decoding. This method significantly boosts responsiveness without compromising accuracy by increasing the hit rate. 

![][image1]

**FireAttention,** Fireworks AI's custom-built inference engine, significantly enhances LLM inference speed on GPUs. It achieves this by utilizing a novel micro-precision data format and rewriting key GPU kernels (such as attention and matrix multiplication) from scratch, aligning them with underlying hardware instructions. While FireAttention prioritizes speed, potentially at the cost of initial accuracy, this is mitigated through Quantization-Aware Training (QAT). This approach allows finetuned models to maintain high precision while reducing their memory footprint. Benchmarks demonstrate FireAttention V4's superior performance over SGLang on H200 and TRT-LLM on B200, particularly in MMLU Pro tests. Overall, FireAttention V4 represents a breakthrough in achieving low-latency, high-efficiency LLM inference, especially beneficial for frontier models like DeepSeek R1.

**Key Benefits:**

* **Faster Inference:** FireOptimizer's adaptive speculative execution has demonstrated up to 3x latency improvements in production workloads across various models, ensuring highly responsive applications.  
* **Hassle-Free Optimization:** FireOptimizer automates the complexities of optimization, allowing users to concentrate on application development.

**FireOptimizer**

* FireOptimizer improves batch inferencing by integrating with MongoDB for efficient model fine-tuning and streamlined deployment. This multi-layered customization is vital for compound AI systems, ensuring consistent model alignment. Available for enterprise on-premise and own-cloud, FireOptimizer enhances traditional inference performance through techniques like **adaptive speculative execution**, **caching**, **customizable quantization**, and **personalized fine-tuning** at scale, along with customizable hardware mapping. In this blog we are going to explore the benefits of FireOptimizer to perform **Parameter-Efficient Fine-Tuning(PEFT)** so we can use a  Small Language Model(SLM) model to carry out personalized tasks such as RAG for private dataset. This activity will demonstrate how generative AI can be adopted for general use at scale and critical domain effectively.

## **Survey of Fine-Tuning Strategies for Smaller, Efficient Models**

Smaller language models present significant opportunities for tailored adaptation while using fewer resources. The ongoing evolution in this field is fueled by increasing demand for deploying optimized LLMs across diverse environments, including cloud platforms, edge devices, and specialized hardware. These fine-tuning approaches can be categorized as follows:

1. **Additive Parameter-Efficient Fine-Tuning (PEFT):** This class of methods augments pre-trained models with new trainable parameters without altering the original weights.  
* **Adapters**: These involve inserting small, trainable modules, known as adapters, within the pre-trained model's layers. These adapters learn task-specific adjustments, enabling adaptation to new tasks without changing the pre-existing parameters.  
* **Soft Prompts**: These are trainable vector embeddings appended to the input sequence, acting as guiding signals to influence the model's output for a specific task.  
* **Prefix Tuning**: This technique adds a trainable prefix to the input sequence. This prefix learns task-specific information without requiring modifications to the core model architecture.  
2. **Reparametrization PEFT**: This approach reduces the number of trainable parameters by reparameterizing existing model weights using low-rank approximations.  
* **Low-Rank Adaptation (LoRA)**: LoRA approximates weight updates in the attention layers of a pre-trained model using low-rank matrices, significantly decreasing the number of trainable parameters.  
* **Quantized LoRA (QLoRA)**: QLoRA builds upon LoRA by integrating quantization methods, further decreasing memory footprint and computational expenses.  
3. **Selective Fine-Tuning**: This category focuses on fine-tuning only specific parameters of the pre-trained model, leading to improved computational efficiency.  
* **BitFit**: This method fine-tunes only the bias terms, or other designated parameters, of the pre-trained model, enhancing computational efficiency.  
* **DiffPruning**: This technique identifies and removes parameters that have minimal impact on the model's performance, thus reducing the number of trainable parameters.  
4. **Layer Freezing Strategies**: These strategies involve selectively freezing certain layers of the pre-trained model while fine-tuning others to optimize the adaptation process.  
* **Freeze and Reconfigure (FAR)**: FAR involves freezing specific layers of the pre-trained model and fine-tuning the remaining layers to optimize model adaptation.  
* **FishMask**: This technique uses a mask to selectively freeze or fine-tune layers, optimizing adaptation for specific tasks..

Parameter-Efficient Fine-Tuning (PEFT) is a popular technique for adapting small pre-trained models to niche tasks. By adjusting only a small portion of the model's parameters, PEFT prevents overfitting, especially on smaller datasets, and greatly reduces computational and memory demands compared to full fine-tuning. Additionally, PEFT helps mitigate catastrophic forgetting in LLMs. This approach allows for efficient model customization in resource-constrained environments without the need for complete retraining.

Leveraging **PEFT LoRA techniques in Fireworks.ai** , combined with availability of trace data and labeled data, allows for efficient fine-tuning of smaller models. 

To demonstrate the **practical implication of using a Small Language Model (SLM)**, we will build an agentic RAG application using **MongoDB Atlas** and illustrate how MongoDB can be used to power semantic search capabilities and also be leveraged as a semantic caching layer. The application serves as a demonstration to follow along with a step by step guide to build a simple application that is task driven by using a Frontier LLM model such as Llama Maverick and they fine tune using data generate out of this setting to fine tine a SLM that will satisfactorily perform a similar operation yet consuming lesser resource.

## **Step-by-Step guide for building an Agentic RAG application with MongoDB Atlas.**

The sample code below demonstrates an end-to-end Agentic Retrieval-Augmented Generation (RAG) workflow using LangChain, MongoDB Atlas Vector Search, and Fireworks LLMs. Below is a summary of the key steps and components:

## 1\. Data Loading & Preprocessing

- **PDF Loading:** The EU Act regulations PDF is loaded using `PyPDFLoader`.  
- **Text Splitting:** The document is split into manageable chunks using `RecursiveCharacterTextSplitter` for efficient retrieval and embedding.

## 2\. Embedding & Vector Store Setup

- **Embeddings:** Sentence-transformers MPNet model is used to generate vector embeddings for each text chunk.  
- **MongoDB Atlas Vector Search:** The embeddings and text chunks are stored in MongoDB, and a vector search index is created for similarity search.

## 3\. LLM & Caching

- **LLM Setup:** Meta Llama Maverick is used as the main LLM, with a custom output parser to clean up responses.  
- **Semantic Cache:** MongoDB Atlas Semantic Cache is configured to cache LLM responses and avoid redundant computation.

## 4\. Agentic RAG Workflow

- **StateGraph Construction:** The workflow is modeled as a state machine with the following steps:  
  - **plan\_step:** Reformulates the user query for optimal retrieval.  
  - **retrieve\_documents\_step:** Retrieves relevant documents from the vector store.  
  - **execute\_step:** Generates an answer using the LLM and retrieved context.  
  - **validate\_step:** Uses the LLM to validate the relevance of the answer.  
  - **should\_continue:** Decides whether to proceed to the execute step or go back to the plan step.

## **Running the code and initial setup**

### 1\. Prerequisites

- **Python 3.9+** (recommended: 3.10 or 3.11)  
- **MongoDB Atlas account** (free tier is sufficient)  
- **Fireworks.ai API key** (for LLM access)  
- **Jupyter Notebook** installed

### 2\. Create and Activate a Virtual Environment

```shell
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3\. Install Required Packages

```shell
pip install jupyterlab
pip install langchain langchain-community langchain-mongodb langchain-fireworks pymongo certifi python-dotenv sentence-transformers
```

### 4\. Set Up Environment Variables

Create a `.env` file in your project directory with the following content:

```
MONGODB_URI=your_mongodb_atlas_connection_string
FIREWORKS_MODEL_ID=accounts/fireworks/models/llama-v3-8b-instruct
FIREWORKS_API_KEY=your_fireworks_api_key
```

- Replace `your_mongodb_atlas_connection_string` with your MongoDB Atlas URI.  
- Replace `your_fireworks_api_key` with your Fireworks.ai API key.

### 5\. Start Jupyter Notebook

Create an empty jupyter notebook and place your PDF (e.g., `eu_act_regulations.pdf`) in the same directory.  
To create a Jupyter Notebook, first ensure JupyterLab is installed using \`pip install jupyterlab\`. Then, open a terminal or command prompt, navigate to your desired directory, and start JupyterLab with the command \`jupyter lab\`, which will open in your web browser. In JupyterLab, either find the "Notebook" section in the launcher and select a kernel (like Python) or go to "File" \> "New" \> "Notebook" to create a new .ipynb file, which you can rename by right-clicking the tab or double-clicking the title.

```shell
jupyter lab
```

Open the notebook in your browser.

### 6\. Run the Notebook Cells

- Execute each cell in order, from top to bottom as described in the   
- The notebook will load your PDF, split it, embed the chunks, and set up the vector store and LLM.  
- You can then run queries using the provided workflow.

---

**Note:**

- Ensure your MongoDB Atlas cluster is running and accessible.  
- The first run may take a few minutes as embeddings are generated and stored.  
- For best results, use a GPU-enabled machine for local embedding, or use a managed embedding service.

## **Code Snippets explained**

Execute the code snippets sequentially in a Jupyter Notebook environment by copy pasting the code snippets below to run the Agentic Retrieval-Augmented Generation (RAG) workflow. Begin by loading and preprocessing the PDF document, then initialize the embedding model and set up the MongoDB Atlas vector store. Next, configure the LLM and semantic cache. Proceed to define the workflow steps as functions for planning, retrieval, execution, and validation. Subsequently, build and compile the workflow graph and invoke it with a user query. Ensure all environment variables, such as MongoDB URI, Fireworks API key and Fireworks Model ID, are correctly set in a \`.env\` file before execution.

Sample \`.env\` 

```
MONGODB_URI=
FIREWORKS_API_KEY=
FIREWORKS_MODEL_ID=
```

### 1\. Data Loading & Preprocessing

**Load the PDF document:**

```py
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("eu_act_regulations.pdf")
docs = loader.load()
```

**Split the document into manageable text chunks:**

```py
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
texts = text_splitter.split_documents(docs)
```

---

### 2\. Embedding & Vector Store Setup

**Initialize the embedding model:**

```py
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
EMBED_DIMENSION = len(embeddings.embed_query("test"))
```

**Connect to MongoDB and set up the vector store:**

```py
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv
load_dotenv()
import os

client = MongoClient(os.getenv("MONGODB_URI"), tlsCAFile=certifi.where())
db = client["agenticrag"]
collection = db["fireworks"]

from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
vectorstore = MongoDBAtlasVectorSearch(
    collection,
    embedding=embeddings,
    index_name="default"
)
vectorstore.add_documents(texts)
```

**Create and validate the vector search index:**

```py
from pymongo.operations import SearchIndexModel

def get_index_config(index_name):
    idxs = list(collection.list_search_indexes())
    for ele in idxs:
        if ele["name"]==index_name:
            return ele

index = SearchIndexModel(
    name="default",
    type="vectorSearch",
    definition={
        "fields": [
            {
                "path": "embedding",
                "type": "vector",
                "numDimensions": EMBED_DIMENSION,
                "similarity": "cosine"
            }
        ]
    }
)
if not get_index_config("default"):
    collection.create_search_index(index)

while True: 
    idx = get_index_config("default")
    if idx["queryable"]:
        print("Vector search index created successfully.")
        break
```

---

### 3\. LLM & Caching

**Set up the LLM and semantic cache:**

```py
from langchain_mongodb.cache import MongoDBAtlasSemanticCache
from langchain_core.globals import set_llm_cache 
from langchain_fireworks import ChatFireworks
from langchain_core.output_parsers import StrOutputParser
import re

llm_cache = MongoDBAtlasSemanticCache(
    connection_string=os.getenv("MONGODB_URI"),
    database_name="agenticrag",
    collection_name="cache",
    embedding=embeddings,
    input_key="prompt",
    output_key="response",
    index_name="cache",
    score_threshold=0.95
)
set_llm_cache(llm_cache)

remove_think_tags = lambda text: re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
llm = (
    ChatFireworks(model=os.getenv("FIREWORKS_MODEL_ID"), temperature=0.2)
    | StrOutputParser()
    | (lambda text: {"response": remove_think_tags(text)})
)
```

---

### 4\. Agentic RAG Workflow Construction

**Define workflow steps as functions:**

```py
# Planning function: use LLM to generate a retrieval question from user query
def plan_step(state):
    if "previous_step" in state and state["previous_step"] == "validate_step":
        # If the previous step was validation, use the original question
        user_question = state["question"] + "\n\n" + "Incorrect Answer: "+ state["answer"]
    else:
        user_question = state["question"]
    # Use LLM to rephrase or generate a retrieval question
    prompt = (
        f"Given the following user question, generate a concise and effective retrieval question for searching a document database. "
        f"Only output the retrieval question.\n\nUser question: {user_question}\nRetrieval question:"
    )
    retrieval_response = llm.invoke(prompt)
    # retrieval_question = retrieval_response.content if hasattr(retrieval_response, "content") else str(retrieval_response)
    retrieval_question = retrieval_response["response"] if "response" in retrieval_response else str(retrieval_response)
    state["retrieval_question"] = retrieval_question.strip()
    # state["plan"] = "Generated retrieval question using LLM."
    state["plan"] = f"Generated retrieval question: \"{retrieval_question.strip()}\" using LLM."
    state["previous_step"] = "plan_step"
    return state

# Custom retrieval function
def retrieve_documents_step(state):
    question = state.get("retrieval_question", state["question"])
    docs = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 25, "lambda_mult": 0.75}
    ).get_relevant_documents(question)
    state["docs"] = docs
    state["previous_step"] = "retrieve_documents_step"
    return state

# Execution function: generate answer using LLM and context (no caching)
def execute_step(state):
    docs = state.get("docs", [])
    question = state["question"]
    context = "\n".join(doc.page_content for doc in docs)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    response_obj = llm.invoke(prompt)
    # response = response_obj.content if hasattr(response_obj, 'content') else response_obj
    response = response_obj["response"] if "response" in response_obj else response_obj

    state["answer"] = response
    state["previous_step"] = "execute_step"
    return state

# Validation function: use LLM to check if answer is relevant to the question (score 0-1)
def validate_step(state):
    answer = state.get("answer", "")
    question = state.get("question", "")
    state["previous_step"] = "validate_step"
    if not answer or not question:
        state["valid"] = False
        state["score"] = 0.0        
        return state

    # Use LLM to score relevance
    validation_prompt = (
        f"Given the following question and answer, rate the relevance of the answer to the question on a scale from 0 (not relevant) to 1 (fully relevant). "
        f"Respond with only a float number between 0 and 1.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
    )
    validation_response = llm.invoke(validation_prompt)
    print(f"Validation response: {validation_response}")
    # Extract score from LLM response
    try:
        # score_str = validation_response.content if hasattr(validation_response, "content") else str(validation_response)
        score_str = validation_response["response"] if "response" in validation_response else str(validation_response)
        score = float(score_str.strip().split()[0])
    except Exception:
        score = 0.0

    state["score"] = score
    state["valid"] = score >= 0.7  # threshold for acceptance
    return state

def should_continue(state):
    # Check if the answer is valid
    if state.get("valid", False):
        return END
    else:
        return "plan_step"  # Go back to planning if not valid

```

**Build and compile the workflow graph:**

```py
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

class AgenticRAGState(TypedDict):
    question: str
    retrieval_question: str = None
    previous_step: str = None
    plan: str = None
    docs: list = None
    answer: str = None
    valid: bool = None
    score: float = None

graph = StateGraph(AgenticRAGState)
graph.add_node("plan_step", plan_step)
graph.add_node("retrieve_step", retrieve_documents_step)
graph.add_node("execute_step", execute_step)
graph.add_node("validate_step", validate_step)
graph.set_entry_point("plan_step")
graph.add_edge("plan_step", "retrieve_step")
graph.add_edge("retrieve_step", "execute_step")
graph.add_edge("execute_step", "validate_step")
graph.add_conditional_edges("validate_step", should_continue, {END: END, "plan_step": "plan_step"})
app = graph.compile()
```

Once built the graph for your agentic workflow looks like as shown in Fig.1.  
![][image2]  
Fig 1: Agentic RAG workflow graph

---

### 5\. Running the Agentic RAG Workflow

**Invoke the workflow with a user query:**

```py
query = "In EU act what are the various biometric categorization of data?"
app.invoke({"question": query}, config={"recursion_limit": 30})
```

**Response:**

```
Response: 
In EU legislation, biometric data is categorized and regulated under several key frameworks, primarily focusing on its use, protection, and specific applications. Based on the provided context and relevant regulations:

### 1. **Biometric Verification (Authentication)**  
   - **Definition**: Automated one-to-one comparison of biometric data (e.g., fingerprints, facial recognition) to verify an individual’s identity against previously stored data.  
   - **Purpose**: Authentication (e.g., unlocking devices, accessing services).  
   - **Legal Reference**: Article 3(36) of the cited regulations.

### 2. **Biometric Identification (Matching)**  
   - **Definition**: One-to-many comparison of biometric data against databases to establish identity (e.g., border control, law enforcement).  
   - **Examples**:  
     - **Eurodac** (Regulation 2024/1358): Compares biometric data (e.g., fingerprints) to identify illegally staying third-country nationals or stateless persons.  
     - **Law Enforcement**: Used to locate suspects, victims of trafficking, or missing persons under strict conditions (Article 3(38), 3(39), and provisions like point (i)–(iii)).  

### 3. **Special Categories of Personal Data**  
   - **Status**: Biometric data is classified as a "special category" under:  
     - **GDPR (Article 9(1), Regulation (EU) 2016/679)**: Requires enhanced protections due to its sensitivity.  
     - **Directive (EU) 2016/680** and **Regulation (EU) 2018/1725**: Extend these protections to law enforcement and EU institutions.  
   - **Safeguards**: Pseudonymization, strict access controls, confidentiality obligations, and mandatory deletion after retention periods (points (c)–(e) in the context).  

### 4. **Operational and Sensitive Data**  
   - **Sensitive Operational Data**: Biometric data used in criminal investigations or counter-terrorism, where disclosure could jeopardize proceedings (Article 3(38)).  
   - **Emotion Recognition Systems**: While not explicitly labeled as biometric, these systems infer emotions/intentions (Article 3(39)) and may intersect with biometric processing if tied to identifiable individuals.  

### 5. **Law Enforcement Exceptions**  
   - Biometric data may be processed for:  
     - Preventing terrorist attacks or imminent threats (point (ii)).  
     - Investigating serious crimes (punishable by ≥4 years’ imprisonment) under Annex II (point (iii)).  

### Key Requirements:  
   - **Security**: State-of-the-art measures, pseudonymization, and access documentation (point (c)).  
   - **Restrictions**: Prohibition on unauthorized transfers (point (d)).  
   - **Retention**: Deletion after correcting bias or reaching retention limits (point (e)).  

These categorizations ensure biometric data is used proportionally, with stringent safeguards to protect privacy and fundamental rights under EU law.

```

**Validation Score:**

```
Score: 0.9
```

This notebook provides a modular, agentic RAG pipeline that can be adapted for various document retrieval and question-answering tasks using MongoDB and LLMs.

## **Step-by-Step guide for Fine-Tuning a Small Language Model with Fireworks.AI** 

### **Current Challenges with Frontier Models**

The large language model used in the preceding example, `accounts/fireworks/models/deepseek-r1`, can result in slow application response times due to the significant computational resources required for its billions of parameters. An agentic RAG task involves multiple LLM invocations for steps such as generating retrieval questions, producing answers, and comparing user questions to the generated results. This process involves several LLM queries, extending the total response time to **30-40 seconds**, with each query potentially taking **5 or more seconds**. Additionally, deploying and scaling LLMs for a large user base can be complex and expensive. To mitigate this issue, the example code demonstrates the use of a semantic cache; however, this only addresses repeated queries to the system.

By leveraging Small Language Models (SLMs), enterprises can achieve significant gains in processing speed and cost-efficiency. SLMs require less computational power, making them ideal for resource-constrained devices, while delivering faster response times and lower operational costs. But there is a huge caveat using SLM, they come with several limitations, such as reduced generalization, limited context retention, and lower accuracy on complex tasks compared to larger models. They may struggle with nuanced reasoning, exhibit increased biases, and generate hallucinations due to their constrained training data and fewer parameters. While they are computationally efficient and well-suited for lightweight applications, their ability to adapt across domains remains restricted, for example a pretrained SLM such as `accounts/fireworks/models/deepseek-r1-distill-qwen-1p5b` does not produce results which is satisfactorily in our agentic RAG setting. It is not able perform validation scoring or tends to hallucinate which generates response even when context is provided.

Adapting a pre-trained Small Language Model (SLM) for specialized applications such as agentic Retrieval-Augmented Generation (RAG) utilizing private knowledge bases offers a cost-effective alternative to frontier models while maintaining similar performance levels. This strategy also provides scalability for numerous clients, ensuring Service Level Agreements (SLAs) are met.  
**Parameter-Efficient Fine-Tuning(PEFT) i.e. QLoRA** techniques, including Quantized Low-Rank Adaptation (LoRA), substantially improve efficiency by focusing optimization on a limited set of parameters. This method lowers memory demands and operational expenses. Integrating with MongoDB streamlines data management and supports efficient model fine-tuning workflows.

### **MongoDB's Unique Value**

MongoDB is integral, providing seamless data management and real-time integration that improves operational efficiency. By storing trace data as JSON and enabling efficient retrieval and storage, MongoDB adds substantial value to the process of fine-tuning models. MongoDB also doubles up as a caching layer to avoid unnecessarily invoking LLM on repeated requests for the same data.

## The following steps will go through step-by-step, how one can make use of the platform to fine tune a SLM. 

Here’s how to leverage this platform and tool:

![][image3]

Fig 2: Fine tuning process explained

To enhance RAG applications, the initial step involves collecting data relevant to the specific task for fine-tuning. MongoDB Atlas, a flexible database, can be utilized to store LLM responses in a cache. For example, in our agentic RAG approach, we can create questions using diverse datasets and store their corresponding answers in MongoDB Atlas. While a powerful LLM might be useful for generating these initial responses or task-specific data during this simulation phase, a smaller scale fine-tuning process requires at least 1000 examples.

Subsequently, these generated responses need to be converted into the required format for the Fireworks.ai platform to begin the fine-tuning process. The `cache.jsonl` file, used later in fine-tuning, can be created by executing the provided code.

```py
from pymongo import MongoClient
import pandas as pd
import json

client = MongoClient("<mongodb_atlas_connection_string>")
cache_col = client["agenticrag"]["cache"]
df = pd.DataFrame.from_records(cache_col.find())
vals = list(zip([{"role": "user", "content": json.loads(text)[0]["kwargs"]["content"]} for text in df.text], [
            {"role": "assistant", "content": json.loads(json.loads(text)[0])["kwargs"]["text"]} for text in df.return_val]))
messages = []
for val in vals:
    messages += [{"messages": list(val)}]
with open("cache.jsonl", "w") as f:
    for item in messages:
        f.write(json.dumps(item) + "\n")
```

Now that we have prepared the dataset and generated our `cache.jsonl` file, we can fine-tune the pre-trained `deepseek-r1-distill-qwen-1p5b` model by following the steps below.

**Prerequisites:**

* **Install firectl:** Use the command `pip install firectl` to install the Fireworks command-line tool.  
* **Authenticate:** Log in to your Fireworks account using `firectl login`.  
* **Prepare Dataset:** Ensure your fine-tuning dataset (created during the data generation process) is ready.

**Steps:**

1. **Upload Dataset:** Upload your prepared dataset to the Fireworks platform using the following command, replacing \`\<dataset\_name\>\` with your desired name and \`cache.jsonl\` with your dataset file:

```
firectl create dataset <dataset_name> cache.jsonl
```

3. **Create Fine-Tuning Job:** Initiate a fine-tuning job by specifying the base model, dataset, output model name, LoRA rank, and number of epochs. For example:

```
firectl create sftj --base-model accounts/fireworks/models/deepseek-r1-distill-qwen-1p5b \
 --dataset <dataset_name> --output-model ragmodel --lora-rank 8 --epochs 1
```

6. The output will provide details about the job, including its name, creation time, dataset used, current state, and the name of the output model.  
7. **Monitor Fine-Tuning:** Track the progress of your fine-tuning job using the Fireworks AI portal. This allows you to ensure the process is running as expected.  
8. **Deploy Fine-Tuned Model:** Once the fine-tuning is complete, deploy the model for inference on the Fireworks platform. This involves two steps:  
   Deploy the base model used for fine-tuning:

```
firectl create deployment accounts/fireworks/models/deepseek-r1-distill-qwen-1p5b --enable-addons --wait
```

   

   Deploy the fine-tuned LoRA adapter:

```
firectl load-lora ragmodel --deployment  <deployment_id>
```

9. **Use Deployed Model:** After deployment, the model ID (e.g., \`models/ragmodel\`) can be used to invoke the fine-tuned language model via your preferred LLM framework, leveraging the Fireworks platform's serverless API.

The integration between MongoDB and Fireworks AI offers a streamlined and efficient approach to enhance AI model performance for RAG applications, as demonstrated by this fine-tuning process.

## **Summary**

Fine-tuning smaller language models (SLMs) for Retrieval Augmented Generation (RAG) using platforms like Fireworks AI offers significant advantages over relying solely on large frontier models. This approach drastically improves response times, reducing latency from around **5 seconds with a large LLM to 2.3 seconds with a fine-tuned SLM**, while also substantially decreasing memory and hardware requirements. By leveraging parameter-efficient fine-tuning techniques and integrating with data management solutions like MongoDB, businesses can achieve faster, more cost-effective AI performance for RAG applications, making advanced AI capabilities more accessible and sustainable.

## **Conclusion**

The collaboration between MongoDB and Fireworks AI offers a powerful synergy for enhancing the efficiency and affordability of Large Language Model (LLM) training and deployment. Fireworks AI's utilization of Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA and qLoRA significantly curtails the computational resources necessary for fine-tuning LLMs by focusing on low-rank adaptation and quantization. This directly translates to substantial reductions in the costs associated with this crucial process. Complementarily, MongoDB's robust infrastructure, characterized by its distributed architecture, flexible schema, and efficient indexing capabilities, provides the ideal data management foundation. It allows for on-demand scaling of data infrastructure while minimizing storage expenses, thereby contributing to lower capital and operational expenditures.

This integration further fosters streamlined workflows between data and AI processes. MongoDB's capacity for real-time data integration ensures that AI models have immediate access to the most current information, thereby improving operational efficiency and the relevance of the models' insights. When combined with Fireworks AI's fine-tuning tools, this creates a cohesive environment where AI models can be continuously updated and refined. Moreover, the partnership simplifies the development of robust Retrieval Augmented Generation (RAG) solutions. MongoDB Atlas offers a scalable platform for storing embeddings, while Fireworks AI provides managed LLM hosting and other essential features. This seamless combination enables the creation of scalable and intelligent systems that significantly enhance user experience through more effective and relevant information retrieval.

Organizations adopting this strategy can achieve accelerated AI performance, resource savings, and future-proof solutions—driving innovation and competitive advantage across different sectors.

## Further Reading

* **FireAttention V2:** Improves long-context inference efficiency by up to 12x. [https://fireworks.ai/blog/fireattention-v2-long-context-inference](https://fireworks.ai/blog/fireattention-v2-long-context-inference)  
* **FireAttention V3:** Enables cost-effective GPU inference with AMD hardware. [https://fireworks.ai/blog/fireattention-v3](https://fireworks.ai/blog/fireattention-v3)  
* **FireOptimizer:** Allows users to customize latency and quality for production inference workloads. [https://fireworks.ai/blog/fireoptimizer](https://fireworks.ai/blog/fireoptimizer)

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,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>

[image3]: <data:image/png;base64,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>