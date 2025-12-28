import pandas as pd
from openai import OpenAI
from pinecone import Pinecone
from tqdm import tqdm

from constants import CHUNK_SIZE, OVERLAP_RATIO, PINECONE_API, MODEL_API


def chunk_text(transcript, chunk_size=CHUNK_SIZE, overlap=OVERLAP_RATIO):
    step = int(chunk_size * (1 - overlap))
    chunks_lst = []
    for i in range(0, len(transcript), step):
        chunks_lst.append(transcript[i:i + step])
    return chunks_lst


def upload_embeds(texts, metadatas):
    responses = client.embeddings.create(input=texts, model='RPRTHPB-text-embedding-3-small')
    vectors = [{"id": metadatas[eid]["id"],
                "values": embedding.embedding,
                "metadata": metadatas[eid]["metadata"]} for eid, embedding in enumerate(responses.data)]
    return vectors


if __name__ == '__main__':
    client = OpenAI(api_key=MODEL_API, base_url="https://api.llmod.ai")
    pc = Pinecone(api_key=PINECONE_API)
    index = pc.Index("ted-talks")

    df = pd.read_csv("ted_talks_en.csv")

    texts, metadatas = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        chunks = chunk_text(row["transcript"])

        for cid, chunk in enumerate(chunks):
            texts.append(f"Title: {row['title']} Speaker: {row['speaker_1']}\n{chunk}")
            metadatas.append({
                "id": f"{row['talk_id']}_{cid}",
                "metadata": {
                    "talk_id": str(row["talk_id"]),
                    "title": row["title"],
                    "speaker": row["speaker_1"],
                    "chunk": chunk
                }
            })

            # embed and upload in batches
            if len(texts) > 100:
                vectors = upload_embeds(texts, metadatas)
                index.upsert(vectors)
                texts, metadatas = [], []

    if texts:
        vectors = upload_embeds(texts, metadatas)
        index.upsert(vectors)
