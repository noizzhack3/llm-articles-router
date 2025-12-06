from pydantic import BaseModel, Field


class DeskDecision(BaseModel):
    """
    Data Transfer Object for a decision made by the news desk regarding an article.
    """
    desk_name: str = Field(description="the name of the desk.")
    should_handle: bool = Field(description="should the desk handle the article or not.")
    reason: str = Field(description="the reason why the desk handle the article or not.")
