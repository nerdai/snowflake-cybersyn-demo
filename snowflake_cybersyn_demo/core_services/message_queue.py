from llama_agents.message_queues.apache_kafka import KafkaMessageQueue

from snowflake_cybersyn_demo.utils import load_from_env

message_queue_host = load_from_env("KAFKA_HOST")
message_queue_port = load_from_env("KAFKA_PORT")

message_queue = KafkaMessageQueue.from_url_params(
    host=message_queue_host,
    port=int(message_queue_port) if message_queue_port else None,
)
