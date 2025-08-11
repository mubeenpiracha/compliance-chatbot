"""
Pydantic models for the new agent-based reasoning architecture.
"""
from typing import List, Union, Optional, Literal
from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    """A single, targeted query to be executed by a search engine."""
    query_text: str = Field(description="The text of the search query.")
    search_type: Literal["vector", "keyword"] = Field(description="The type of search to perform.")
    purpose: str = Field(description="The reason for this specific query (e.g., 'definition of fund').")

class SearchPlan(BaseModel):
    """A comprehensive plan of multiple search queries to execute."""
    search_queries: List[SearchQuery] = Field(description="A list of diverse search queries to run in parallel.")

class ClarificationRequest(BaseModel):
    """A request for more information from the user."""
    clarification_questions: List[str] = Field(description="A list of questions to ask the user to resolve ambiguity.")

class QueryAnalysis(BaseModel):
    """
    The output of the initial query analysis step.
    The system decides whether it can proceed with a search or must ask for clarification.
    """
    decision: Union[SearchPlan, ClarificationRequest] = Field(description="The decision to either create a search plan or request clarification.")
    reasoning: str = Field(description="The reasoning behind the decision.")
