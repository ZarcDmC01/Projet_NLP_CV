import enum
from sqlalchemy import create_engine, Column, String, Boolean, BigInteger, Enum
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./nlp_cv.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UserRole(str, enum.Enum):
    USER = "user"
    ADMIN = "admin"


class AccountStatus(str, enum.Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(UserRole), default=UserRole.USER)
    status = Column(Enum(AccountStatus), default=AccountStatus.ACTIVE)
    email_verified = Column(Boolean, default=False)


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    language = Column(String, default="fr")
    timezone = Column(String, default="UTC")


class StorageQuota(Base):
    __tablename__ = "storage_quotas"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    limit_bytes = Column(BigInteger, default=5 * 1024 * 1024 * 1024)


Base.metadata.create_all(bind=engine)
