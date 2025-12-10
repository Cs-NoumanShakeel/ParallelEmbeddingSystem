import os
import pika
import json
from dotenv import load_dotenv
from src.embedding import EmbeddingPipeline
from src.vectorstore import CHROMAVectorStore

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_QUEUE = os.getenv("RABBITMQ_QUEUE", "pdf_chunks")

# -------------------------------
# Initialize embedding pipeline & vector store
# -------------------------------
print("[Worker] Initializing embedding pipeline and vector store...")
emb_pipeline = EmbeddingPipeline()  # model will be loaded
vector_store = CHROMAVectorStore()
print("[Worker] Initialization complete")

# -------------------------------
# RabbitMQ connection
# -------------------------------
def connect_rabbitmq():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=RABBITMQ_QUEUE, durable=True)
    return connection, channel

# -------------------------------
# Callback for processing batch messages
# -------------------------------
def callback(ch, method, properties, body):
    try:
        # Decode the message
        data = json.loads(body)
        
        # Debug: Print received data structure
        print(f"[Worker] Received message with keys: {list(data.keys())}")
        
        # Validate message structure
        if "texts" not in data:
            raise KeyError("Message missing 'texts' field. Received keys: " + str(list(data.keys())))
        
        texts = data["texts"]
        metadatas = data.get("metadatas", [{}] * len(texts))
        ids = data.get("ids", [f"chunk_{i}" for i in range(len(texts))])
        
        print(f"[Worker] Processing batch with {len(texts)} chunks")

        # Preprocess
        preprocessed_texts = emb_pipeline.preprocess_texts(texts)

        # Generate embeddings
        embeddings = emb_pipeline.embed_texts(preprocessed_texts)

        # Store in vector DB
        vector_store.add_embeddings(
            texts=preprocessed_texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        # Acknowledge message
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f"[Worker] ✓ Successfully processed {len(texts)} chunks")

    except KeyError as e:
        print(f"[Worker] ✗ KeyError: {e}")
        print(f"[Worker] Message content: {body.decode()[:200]}...")  # Print first 200 chars
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    except json.JSONDecodeError as e:
        print(f"[Worker] ✗ JSON decode error: {e}")
        print(f"[Worker] Raw message: {body.decode()[:200]}...")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    except Exception as e:
        print(f"[Worker] ✗ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


# -------------------------------
# Start consuming
# -------------------------------
if __name__ == "__main__":
    print(f"[Worker] Connecting to RabbitMQ at {RABBITMQ_HOST}, queue: {RABBITMQ_QUEUE}")
    print("[Worker] Waiting for tasks. To exit press CTRL+C")
    
    connection, channel = connect_rabbitmq()
    channel.basic_qos(prefetch_count=1)  # fair dispatch
    channel.basic_consume(queue=RABBITMQ_QUEUE, on_message_callback=callback)
    
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("\n[Worker] Interrupted. Closing connection.")
        channel.stop_consuming()
        connection.close()
    except Exception as e:
        print(f"[Worker] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        connection.close()