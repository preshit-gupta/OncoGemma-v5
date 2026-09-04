from fastapi import Header, HTTPException, status
from pydantic import BaseModel
from app.core.config import settings

class CurrentUser(BaseModel):
    id: str
    email: str
    role: str # admin, pathologist, technician, viewer

def get_current_user(
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_user_role: str | None = Header(None, alias="X-User-Role"),
    x_user_email: str | None = Header(None, alias="X-User-Email")
) -> CurrentUser:
    """
    FastAPI dependency for user authentication.
    Supports mock auth headers for dev/testing, extensible to Firebase Auth token verification.
    """
    if settings.MOCK_AUTH_ENABLED:
        user_id = x_user_id or settings.DEFAULT_MOCK_USER_ID
        role = x_user_role or settings.DEFAULT_MOCK_ROLE
        email = x_user_email or f"{user_id}@oncogemma.health"
        
        if role not in ["admin", "pathologist", "technician", "viewer"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Invalid user role: {role}"
            )
            
        return CurrentUser(id=user_id, email=email, role=role)
        
    # Firebase Auth token verification can be plugged in here in later stages
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required"
    )
