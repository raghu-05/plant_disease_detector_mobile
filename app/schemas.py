from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional

# --- ADD THESE NEW MODELS ---
class UserBase(BaseModel):
    username: str
    email: str  
    name: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    model_config = ConfigDict(from_attributes=True)

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
# --- END OF NEW MODELS ---


# Pydantic model for creating a log (used in the POST request)
class DiagnosisLogCreate(BaseModel):
    disease_name: str
    severity: float
    latitude: float
    longitude: float

# Pydantic model for reading a log (used in the GET response)
# This model will be used to convert the SQLAlchemy object to a JSON response.
class DiagnosisLog(BaseModel):
    id: int
    disease_name: str
    severity: float
    latitude: float
    longitude: float
    timestamp: datetime
    owner_id: int

# --- ADD THIS AT THE BOTTOM OF app/schemas.py ---

class FeedbackCreate(BaseModel):
    name: str
    email: str
    message: str
    
    # This config class is the magic that makes it work.
    # It tells Pydantic to read the data even if it's not a dict,
    # but an ORM model (like our SQLAlchemy model).
    #class Config:
    #    orm_mode = True
    model_config = ConfigDict(from_attributes=True)