from fastapi import FastAPI, Request
from openai import OpenAI
from pinecone import Pinecone

from constants import CHUNK_SIZE, OVERLAP_RATIO, TOP_K, MODEL_API, PINECONE_API, SYSTEM_PROMPT

app = FastAPI()

client = OpenAI(api_key=MODEL_API, base_url="https://api.llmod.ai")
pc = Pinecone(api_key=PINECONE_API)
index = pc.Index("ted-talks")


@app.post("/api/prompt")
async def prompt(request: Request):
    data = await request.json()
    question = data.get("question")

    q_response = client.embeddings.create(input=question, model='RPRTHPB-text-embedding-3-small')
    q_embedding = q_response.data[0].embedding

    results = index.query(
        vector=q_embedding,
        top_k=TOP_K,
        include_metadata=True
    )

    context = []
    context_text = ""
    for match in results['matches']:
        meta = match['metadata']
        context.append({
            "talk_id": meta["talk_id"],
            "title": meta["title"],
            "chunk": meta["chunk"],
            "score": match['score']
        })
        context_text += f"\nTitle: {meta['title']}\nSpeaker: {meta.get('speaker', 'Unknown')}\nContent: {meta['chunk']}\n\n"

    usr_prompt = f"Context:\n{context_text}\n\nQuestion: {question}"
    chat_response = client.chat.completions.create(
        model="RPRTHPB-gpt-5-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": usr_prompt}
        ]
    )
    answer = chat_response.choices[0].message.content

    return {
        "response": answer,
        "context": context,
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": usr_prompt
        }
    }


@app.get("/api/stats")
def stats():
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }
