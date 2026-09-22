from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime

from app.database.database import Base
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    JSON
)

from datetime import datetime

class Paper(Base):
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, index=True)

    title = Column(String, nullable=False)

    authors = Column(Text, nullable=True)

    year = Column(Integer, nullable=True)

    abstract = Column(Text, nullable=True)

    doi = Column(String, nullable=True)

    url = Column(String, nullable=True)

    source = Column(String, nullable=True)

    pdf_path = Column(String, nullable=True)

    text_path = Column(String, nullable=True)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )
class PaperAnalysis(Base):
    __tablename__ = "paper_analysis"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    paper_id = Column(
        Integer,
        ForeignKey("papers.id"),
        unique=True,
        nullable=False,
        index=True
    )

    objective = Column(
        JSON,
        nullable=True
    )

    methodology = Column(
        JSON,
        nullable=True
    )

    dataset = Column(
        JSON,
        nullable=True
    )

    model = Column(
        JSON,
        nullable=True
    )

    evaluation_metrics = Column(
        JSON,
        nullable=True
    )

    main_results = Column(
        JSON,
        nullable=True
    )

    limitations = Column(
        JSON,
        nullable=True
    )

    future_work = Column(
        JSON,
        nullable=True
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )