import pika
import json
from src.data_loader import load_docs
from src.embedding import EmbeddingPipeline

# ------------------------
# Configuration
# ------------------------
RABBITMQ_HOST = 'localhost'
RABBITMQ_QUEUE = 'pdf_chunks'
BATCH_SIZE = 10  # number of chunks per message

# ------------------------
# RabbitMQ Publisher
# ------------------------
class Producer:
    def __init__(self, host=RABBITMQ_HOST, queue=RABBITMQ_QUEUE):
        self.host = host
        self.queue = queue
        self._connect()

    def _connect(self):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue, durable=True)
        print(f"[Producer] Connected to RabbitMQ queue '{self.queue}'")

    def publish_batch(self, batch: dict):
        """Publish a batch of chunks"""
        message = json.dumps(batch)
        self.channel.basic_publish(
            exchange='',
            routing_key=self.queue,
            body=message,
            properties=pika.BasicProperties(delivery_mode=2)  # persistent
        )
        # Print first chunk ID for verification
        first_id = batch['ids'][0] if batch.get('ids') else 'unknown'
        print(f"[Producer] Published batch with {len(batch['texts'])} chunks (first: {first_id})")

    def close(self):
        self.connection.close()
        print("[Producer] Connection closed")


# ------------------------
# Controller logic
# ------------------------
def main(data_dir='./data'):
    docs = load_docs(data_dir)
    print(f"[Controller] Loaded {len(docs)} documents")

    emb_pipeline = EmbeddingPipeline(load_model=False)  # controller only chunks
    chunks = emb_pipeline.chunk_documents(docs)

    producer = Producer()
    batch_texts, batch_metadatas, batch_ids = [], [], []

    for i, chunk in enumerate(chunks):
        batch_texts.append(chunk.page_content)
        batch_metadatas.append(chunk.metadata)
        
        # Create a unique ID for each chunk
        source = chunk.metadata.get('source', f'doc_{i}')
        page = chunk.metadata.get('page', 0)
        chunk_id = f"{source}_page{page}_chunk{i}"
        batch_ids.append(chunk_id)

        # Publish when batch is full
        if len(batch_texts) >= BATCH_SIZE:
            producer.publish_batch({
                "texts": batch_texts,
                "metadatas": batch_metadatas,
                "ids": batch_ids
            })
            batch_texts, batch_metadatas, batch_ids = [], [], []

    # Publish remaining chunks
    if batch_texts:
        producer.publish_batch({
            "texts": batch_texts,
            "metadatas": batch_metadatas,
            "ids": batch_ids
        })

    producer.close()
    print(f"[Controller] All {len(chunks)} chunks published to RabbitMQ")


if __name__ == "__main__":
    main()