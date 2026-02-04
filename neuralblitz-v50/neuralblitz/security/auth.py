"""
NeuralBlitz V50 - RBAC & Authentication System
Role-Based Access Control with JWT and API key management.
"""

try:
    from jose import JWTError, jwt
    from passlib.context import CryptContext

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from enum import Enum
import secrets
import hashlib
from dataclasses import dataclass
import threading


class Permission(Enum):
    """Available permissions."""

    READ = "read"
    PROCESS = "process"
    ADMIN = "admin"
    DELETE = "delete"


class Role(Enum):
    """Predefined roles with permission sets."""

    VIEWER = {Permission.READ}
    USER = {Permission.READ, Permission.PROCESS}
    ADMIN = {Permission.READ, Permission.PROCESS, Permission.ADMIN, Permission.DELETE}


@dataclass
class User:
    """User entity."""

    username: str
    role: Role
    api_key: str
    created_at: datetime
    last_accessed: Optional[datetime] = None
    rate_limit: int = 100  # requests per minute


@dataclass
class APIKey:
    """API key entity."""

    key: str
    username: str
    permissions: Set[Permission]
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True


class RBACManager:
    """
    Role-Based Access Control Manager.

    Features:
    - User management
    - API key generation and validation
    - JWT token creation and verification
    - Permission checking
    - Rate limiting per user
    """

    def __init__(self, secret_key: Optional[str] = None):
        if not JWT_AVAILABLE:
            raise ImportError(
                "JWT dependencies not installed. "
                "Install with: pip install python-jose[cryptography] passlib[bcrypt]"
            )

        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        # Storage (in production, use database)
        self._users: Dict[str, User] = {}
        self._api_keys: Dict[str, APIKey] = {}
        self._rate_limit_counters: Dict[str, List[datetime]] = {}
        self._lock = threading.Lock()

    def create_user(
        self, username: str, role: Role = Role.USER, rate_limit: int = 100
    ) -> User:
        """Create a new user."""
        with self._lock:
            if username in self._users:
                raise ValueError(f"User {username} already exists")

            # Generate API key
            api_key = self._generate_api_key()

            user = User(
                username=username,
                role=role,
                api_key=api_key,
                created_at=datetime.utcnow(),
                rate_limit=rate_limit,
            )

            self._users[username] = user

            # Create API key entry
            api_key_entry = APIKey(
                key=api_key,
                username=username,
                permissions=role.value,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(days=365),
            )
            self._api_keys[api_key] = api_key_entry

            return user

    def _generate_api_key(self) -> str:
        """Generate a secure API key."""
        return f"nb_{secrets.token_urlsafe(32)}"

    def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """Validate an API key."""
        with self._lock:
            if api_key not in self._api_keys:
                return None

            key_data = self._api_keys[api_key]

            if not key_data.is_active:
                return None

            if key_data.expires_at and datetime.utcnow() > key_data.expires_at:
                return None

            return key_data

    def check_permission(self, api_key: str, permission: Permission) -> bool:
        """Check if API key has specific permission."""
        key_data = self.validate_api_key(api_key)
        if not key_data:
            return False

        return permission in key_data.permissions

    def check_rate_limit(self, api_key: str) -> bool:
        """
        Check if request is within rate limit.

        Returns:
            True if allowed, False if rate limited
        """
        with self._lock:
            key_data = self.validate_api_key(api_key)
            if not key_data:
                return False

            user = self._users.get(key_data.username)
            if not user:
                return False

            now = datetime.utcnow()
            window_start = now - timedelta(minutes=1)

            # Get requests in last minute
            if api_key not in self._rate_limit_counters:
                self._rate_limit_counters[api_key] = []

            # Clean old entries
            self._rate_limit_counters[api_key] = [
                t for t in self._rate_limit_counters[api_key] if t > window_start
            ]

            # Check limit
            if len(self._rate_limit_counters[api_key]) >= user.rate_limit:
                return False

            # Record request
            self._rate_limit_counters[api_key].append(now)

            return True

    def get_rate_limit_status(self, api_key: str) -> Dict[str, Any]:
        """Get current rate limit status."""
        with self._lock:
            key_data = self.validate_api_key(api_key)
            if not key_data:
                return {"error": "Invalid API key"}

            user = self._users.get(key_data.username)
            if not user:
                return {"error": "User not found"}

            now = datetime.utcnow()
            window_start = now - timedelta(minutes=1)

            recent_requests = len(
                [
                    t
                    for t in self._rate_limit_counters.get(api_key, [])
                    if t > window_start
                ]
            )

            return {
                "limit": user.rate_limit,
                "used": recent_requests,
                "remaining": user.rate_limit - recent_requests,
                "reset_at": (now + timedelta(minutes=1)).isoformat(),
            }

    def create_jwt_token(
        self, username: str, expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT token for user."""
        if expires_delta is None:
            expires_delta = timedelta(hours=24)

        user = self._users.get(username)
        if not user:
            raise ValueError(f"User {username} not found")

        expire = datetime.utcnow() + expires_delta

        payload = {
            "sub": username,
            "role": user.role.name,
            "exp": expire,
            "iat": datetime.utcnow(),
            "permissions": [p.value for p in user.role.value],
        }

        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except JWTError:
            return None

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        with self._lock:
            if api_key in self._api_keys:
                self._api_keys[api_key].is_active = False
                return True
            return False

    def list_user_api_keys(self, username: str) -> List[APIKey]:
        """List all API keys for a user."""
        return [key for key in self._api_keys.values() if key.username == username]

    def get_user(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self._users.get(username)

    def authenticate_request(
        self,
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
        required_permission: Permission = Permission.PROCESS,
    ) -> Dict[str, Any]:
        """
        Authenticate and authorize a request.

        Args:
            api_key: API key for authentication
            jwt_token: JWT token for authentication
            required_permission: Required permission for the operation

        Returns:
            Authentication result
        """
        # Try API key
        if api_key:
            key_data = self.validate_api_key(api_key)
            if not key_data:
                return {"authenticated": False, "error": "Invalid API key"}

            # Check permission
            if required_permission not in key_data.permissions:
                return {
                    "authenticated": True,
                    "authorized": False,
                    "error": f"Insufficient permissions. Required: {required_permission.value}",
                }

            # Check rate limit
            if not self.check_rate_limit(api_key):
                return {
                    "authenticated": True,
                    "authorized": False,
                    "error": "Rate limit exceeded",
                }

            return {
                "authenticated": True,
                "authorized": True,
                "username": key_data.username,
                "permissions": [p.value for p in key_data.permissions],
            }

        # Try JWT
        if jwt_token:
            payload = self.verify_jwt_token(jwt_token)
            if not payload:
                return {"authenticated": False, "error": "Invalid or expired token"}

            # Check permission
            if required_permission.value not in payload.get("permissions", []):
                return {
                    "authenticated": True,
                    "authorized": False,
                    "error": f"Insufficient permissions. Required: {required_permission.value}",
                }

            return {
                "authenticated": True,
                "authorized": True,
                "username": payload["sub"],
                "permissions": payload["permissions"],
            }

        return {"authenticated": False, "error": "No credentials provided"}


class FastAPIAuthMiddleware:
    """
    FastAPI middleware for authentication.

    Usage:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> rbac = RBACManager()
        >>> auth_middleware = FastAPIAuthMiddleware(rbac)
        >>> app.middleware("http")(auth_middleware)
    """

    def __init__(self, rbac: RBACManager):
        self.rbac = rbac

    async def __call__(self, request, call_next):
        """Process request with authentication."""
        from fastapi import HTTPException

        # Skip auth for public endpoints
        if request.url.path in ["/health", "/docs", "/openapi.json"]:
            return await call_next(request)

        # Get credentials
        api_key = request.headers.get("X-API-Key")
        auth_header = request.headers.get("Authorization")

        jwt_token = None
        if auth_header and auth_header.startswith("Bearer "):
            jwt_token = auth_header.split(" ")[1]

        # Determine required permission
        required_permission = (
            Permission.READ if request.method == "GET" else Permission.PROCESS
        )

        # Authenticate
        result = self.rbac.authenticate_request(
            api_key=api_key,
            jwt_token=jwt_token,
            required_permission=required_permission,
        )

        if not result["authenticated"]:
            raise HTTPException(status_code=401, detail=result["error"])

        if not result["authorized"]:
            raise HTTPException(status_code=403, detail=result["error"])

        # Add user info to request state
        request.state.user = result["username"]
        request.state.permissions = result["permissions"]

        return await call_next(request)


def create_rbac_system(secret_key: Optional[str] = None) -> RBACManager:
    """
    Factory function to create RBAC system with default users.

    Returns:
        Configured RBACManager with demo users
    """
    rbac = RBACManager(secret_key)

    # Create default admin user
    rbac.create_user("admin", role=Role.ADMIN, rate_limit=1000)

    # Create demo user
    user = rbac.create_user("demo", role=Role.USER, rate_limit=100)

    print("RBAC System initialized:")
    print(f"  Admin API Key: {rbac._users['admin'].api_key[:20]}...")
    print(f"  Demo API Key: {user.api_key[:20]}...")

    return rbac


# Export
__all__ = [
    "RBACManager",
    "FastAPIAuthMiddleware",
    "Permission",
    "Role",
    "User",
    "APIKey",
    "create_rbac_system",
    "JWT_AVAILABLE",
]
