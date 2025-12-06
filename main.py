from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from dtos.article import Article

load_dotenv()

model = init_chat_model("gpt-4.1-mini")

with open(
        "c:/code_projects/llm-articles-router/prompts/eng/sports_desk_system_prompt.txt", 'r',
        encoding='utf-8') as file:
    sports_desk_system_prompt = file.read()

new_article_message = """
        ##### news article to process #####
        - title: {title}
        - summary: {summary}
        - article_field: {article_field}
        - article_subdomain: {article_subdomain}
        - places: {places}
        - article_body: {article_body}
    """

sports_desk_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", sports_desk_system_prompt),
        ("human", new_article_message)
    ]
)

output_parser = StrOutputParser()

# router_chain = main_prompt | model | output_parser
sports_desk_chain = sports_desk_prompt_template | model | output_parser

article = Article(**{"title": "Exciting Soccer Match in Madrid",
                     "summary": "Real Madrid clinched a thrilling 3-2 victory against Barcelona in the El Clásico match.",
                     "article_body": "In an electrifying El Clásico held in Madrid, Real Madrid defeated Barcelona 3-2 on April 20, 2024. The match saw standout performances from Karim Benzema, who scored twice, and Vinícius Júnior. The victory boosts Real Madrid's chances in the La Liga championship race.",
                     "article_field": "Sports",
                     "article_subdomain": "Soccer",
                     "places": ["Madrid", "Spain"]})

x = article.model_dump()

out = sports_desk_chain.invoke(
    {"title": "Exciting Soccer Match in Madrid",
     "summary": "Real Madrid clinched a thrilling 3-2 victory against Barcelona in the El Clásico match.",
     "article_body": "In an electrifying El Clásico held in Madrid, Real Madrid defeated Barcelona 3-2 on April 20, 2024. The match saw standout performances from Karim Benzema, who scored twice, and Vinícius Júnior. The victory boosts Real Madrid's chances in the La Liga championship race.",
     "article_field": "Sports",
     "article_subdomain": "Soccer",
     "places": ["Madrid", "Spain"]
     })
print(out)
