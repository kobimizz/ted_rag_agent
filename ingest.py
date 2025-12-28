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


if __name__ == '__main__':
    client = OpenAI(api_key=MODEL_API, base_url="https://api.llmod.ai")
    pc = Pinecone(api_key=PINECONE_API)
    index = pc.Index("ted-talks")

    df = pd.read_csv("small_ted.csv")

    for _, row in tqdm(df.iterrows()):
        chunks = chunk_text(row["transcript"])

        for idx, chunk in enumerate(chunks):
            combined_txt = f"Title: {row['title']} Speaker: {row['speaker_1']}\n{chunk}"
            response = client.embeddings.create(input=combined_txt, model='RPRTHPB-text-embedding-3-small')
            embedding = response.data[0].embedding

            index.upsert([{
                "id": f"{row["talk_id"]}_{idx}",
                "values": embedding,
                "metadata": {
                    "talk_id": f"{row["talk_id"]}",
                    "title": row["title"],
                    "speaker": row["speaker_1"],
                    "chunk": chunk
                }
            }])
