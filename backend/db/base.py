# backend/db/base.py
from sqlalchemy.ext.declarative import as_declarative, declared_attr

@as_declarative()
class Base:
    """
Base class for all SQLAlchemy models.
It provides a default __tablename__ generation.
"""
    id: int
    __name__: str

    # to generate tablename from classname
    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower()
