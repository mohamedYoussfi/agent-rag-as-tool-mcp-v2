from mcp.server.mcpserver import MCPServer
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

load_dotenv(override=True)

mcp = MCPServer(name="mcp-server")


web_search_client = TavilySearch()


@mcp.tool()
def get_employee_infos(name: str):
    """Search web for the given query"""
    return {"name": name, "salary": 12000, "seniority": 12}


@mcp.tool()
def search(query: str):
    """Get info about a given query"""
    print(f"query :{query}")
    results = web_search_client.invoke({"query": query})
    return results


texts = [
    "Je m'appelle Mohamed Youssfi, Je suis Professeur en Informatique et Intelligence artificielle",
    "Je travaille à l'ENSET Mohammedia, Université Hassan II de Casablanca",
    "J'ai obtenu mon doctorat d'état en 2015, Mon doctorat de troisième cycle en 19996 et mon diplôme de professeur second cycle en 1993",
    "En plus de l'informatique, je suis patiené par la musique et la culture",
    "J'aime aussi écrire des récits sur ma vie et j'aime la philosophie",
    "Je suis originaire de Ouarzazate, une ville au sud du Maroc",
    "après les études de primaire et le collège à Ouarzazate, j'ai suivi mes études de lycée technique à Marrakech",
    "Après le baccalauréat, j'ai rejoint l'ENSET Mohammedia pendant 4 années d'études pour devenir Professeur de second cycle",
    "J'ai travaillé à l'ENSET depuis 1993 pour y enseigner principalement l'informatique et les sciences de l'ingénieur",
    "En parallèle à mon travailler de professeur, j'ai suivi mes études supérieures à la Facultés des sciences de rabat",
    "où j'ai obtenu mon DEA, Doctorat de troisième cycle puis Doctorat d'Etat dans le domaine des systèmes informatiques parallèles et distribués",
]
vectore_store = FAISS.from_texts(texts=texts, embedding=embedding_model)

retrieval = vectore_store.as_retriever(kwargs={"k": 5})


@mcp.tool()
def rag_tool(query: str) -> str:
    """Get information about my cv (nom, prénom, diplômes, fonction ...)"""
    print("============ ")
    relevant_chunks = retrieval.invoke(query)
    context_list = [d.page_content for d in relevant_chunks]
    context = ".\n ".join(context_list)
    return context


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=24000)
