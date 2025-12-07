import asyncio
import os
from glob import glob
from typing import Any

from aio_pika import connect_robust, Message, DeliveryMode
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

from dtos.article import Article

load_dotenv()

exchange = None


async def init_rabbitmq():
    global exchange

    if exchange is None:
        connection = await connect_robust("amqp://guest:guest@localhost/breaking_bed")
        channel = await connection.channel()
        exchange = await channel.declare_exchange(
            "breaking_bed",
            type="topic",
            durable=True)


model = init_chat_model("gpt-4.1-mini")

new_article_message = """
        ##### news article to process #####
        - title: {title}
        - summary: {summary}
        - article_field: {article_field}
        - article_subdomain: {article_subdomain}
        - places: {places}
        - article_body: {article_body}
    """

article = Article(**{
    "title": "Exciting Soccer Match in Madrid",
    "summary": "Real Madrid clinched a thrilling 3-2 victory against Barcelona in the El Clásico match.",
    "article_body": "In an electrifying El Clásico held in Madrid, Real Madrid defeated Barcelona 3-2 on April 20, 2024. The match saw standout performances from Karim Benzema, who scored twice, and Vinícius Júnior. The victory boosts Real Madrid's chances in the La Liga championship race.",
    "article_field": "Sports",
    "article_subdomain": "Soccer",
    "places": ["Madrid", "Spain"]
})


async def handle_articles(data: Any):
    print(data)
    return data


async def send_article_to_queue(article: Article):
    """
    send an article object to a queue
    """
    await init_rabbitmq()
    print("\n--- Tool Execution: send_article_to_queue ---")
    print(f"Article:\n {article.model_dump()}")
    print("Sending article object to queue successful.")
    print("------------------------------------------\n")

    message = Message(
        body=article.model_dump_json().encode(),
        delivery_mode=DeliveryMode.PERSISTENT,  # Make the message durable
        content_type='application/json'  # Inform consumers about content type
    )

    await exchange.publish(
        message,
        routing_key="test"
    )

    return "Article successfully sent to the queue."


async def main():
    result = await main_processing_chain.ainvoke(
        {
            "title": "Exciting Soccer Match in Madrid",
            "summary": "Real Madrid clinched a thrilling 3-2 victory against Barcelona in the El Clásico match.",
            "article_body": "In an electrifying El Clásico held in Madrid, Real Madrid defeated Barcelona 3-2 on April 20, 2024. The match saw standout performances from Karim Benzema, who scored twice, and Vinícius Júnior. The victory boosts Real Madrid's chances in the La Liga championship race.",
            "article_field": "Sports",
            "article_subdomain": "Soccer",
            "places": ["Madrid", "Spain"]
        })

    print(result)


def init_llm_pipeline_for_topic(desk_prompt: str):
    desk_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", desk_prompt),
            ("human", new_article_message)
        ]
    )

    str_output_parser = StrOutputParser()

    return desk_prompt_template | model | str_output_parser


def init_llm_pipeline():
    prompt_files = glob("C:\code_projects\llm-articles-router\prompts\eng\*.txt")

    desks_llm_pipelines: dict[str, Any] = {
        "article": RunnablePassthrough()
    }

    for prompt_file in prompt_files:
        desk_topic = os.path.basename(prompt_file).split('_')[0]

        with open(
                prompt_file,
                'r',
                encoding='utf-8') as file:
            desk_prompt = file.read()

            llm_pipeline_for_topic = init_llm_pipeline_for_topic(desk_prompt)
            desks_llm_pipelines[f'{desk_topic}_desk'] = llm_pipeline_for_topic

    parallel_processing_chain = RunnableParallel(
        **desks_llm_pipelines
    )

    send_results = RunnableLambda(handle_articles)

    result = parallel_processing_chain | send_results

    return result


main_processing_chain = init_llm_pipeline()

if __name__ == "__main__":
    asyncio.run(main())
