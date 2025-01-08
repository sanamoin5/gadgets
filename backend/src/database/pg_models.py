from sqlalchemy import (
    Column,
    ForeignKey,
    String,
    DateTime,
    Text,
    Float,
    Integer,
    Boolean,
    Table,
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid
import datetime

Base = declarative_base()

# Many-to-Many Association Table
gadget_category_association = Table(
    "gadget_categories",
    Base.metadata,
    Column("gadget_id", UUID(as_uuid=True), ForeignKey("gadgets.id", ondelete="CASCADE"), primary_key=True),
    Column("category_id", UUID(as_uuid=True), ForeignKey("categories.id", ondelete="CASCADE"), primary_key=True),
)


class User(Base):
    """Represents a user in the system."""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.now(), nullable=False)

    quiz_responses = relationship("QuizResponse", back_populates="user")
    recommendations = relationship("Recommendation", back_populates="user")


class Gadget(Base):
    """Represents a gadget available in the system."""
    __tablename__ = "gadgets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    price = Column(Float, nullable=True)  # Allow null if price is unavailable
    currency = Column(String(10), nullable=True)
    rating = Column(Float, nullable=True)
    reviews_count = Column(Integer, nullable=True)
    image_url = Column(Text)
    amazon_choice = Column(Boolean, default=False)
    best_seller = Column(Boolean, default=False)
    sales_volume = Column(String(255), nullable=True)  # Sales data as text
    shipping_info = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.now(), nullable=False)

    recommendations = relationship("Recommendation", back_populates="gadget")
    categories = relationship("Category", secondary=gadget_category_association, back_populates="gadgets")


class Category(Base):
    """Represents a category for gadgets (e.g., 'xbox console', 'headphones')."""
    __tablename__ = "categories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), unique=True, nullable=False)  # Categories should be unique
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.now(), nullable=False)

    gadgets = relationship("Gadget", secondary=gadget_category_association, back_populates="categories")


class QuizQuestion(Base):
    """Represents a quiz question."""
    __tablename__ = "quiz_questions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question = Column(Text, nullable=False)
    is_main_question = Column(Boolean, default=True, nullable=False)
    interests = Column(Text, nullable=True)  # Comma-separated interests data for mapping
    categories = Column(Text, nullable=True)  # Comma-separated categories for mapping
    created_at = Column(DateTime, default=datetime.datetime.now(), nullable=False)

    options = relationship("QuizOption", back_populates="question")


class QuizOption(Base):
    """Represents a quiz option for a question."""
    __tablename__ = "quiz_options"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question_id = Column(UUID(as_uuid=True), ForeignKey("quiz_questions.id", ondelete="CASCADE"), nullable=False)
    option_text = Column(Text, nullable=False)

    question = relationship("QuizQuestion", back_populates="options")


class QuizResponse(Base):
    """Represents a user's response to a quiz question."""
    __tablename__ = "quiz_responses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    question_id = Column(UUID(as_uuid=True), ForeignKey("quiz_questions.id", ondelete="CASCADE"), nullable=False)
    selected_option = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.now(), nullable=False)

    user = relationship("User", back_populates="quiz_responses")
    question = relationship("QuizQuestion")


class Recommendation(Base):
    """Represents a recommended gadget for a user."""
    __tablename__ = "recommendations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    gadget_id = Column(UUID(as_uuid=True), ForeignKey("gadgets.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.now(), nullable=False)

    user = relationship("User", back_populates="recommendations")
    gadget = relationship("Gadget", back_populates="recommendations")
