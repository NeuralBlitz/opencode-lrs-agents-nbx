"""
Tests for NeuralBlitz RBAC & Authentication System
"""

import pytest
import time
from datetime import datetime, timedelta
from neuralblitz.security.auth import (
    RBACManager,
    Permission,
    Role,
    User,
    APIKey,
    create_rbac_system,
    FastAPIAuthMiddleware,
    JWT_AVAILABLE,
)


class TestRBACManager:
    """Test suite for RBAC functionality."""

    def setup_method(self):
        """Setup RBAC manager for testing."""
        if not JWT_AVAILABLE:
            pytest.skip("JWT dependencies not available")

        self.rbac = RBACManager(secret_key="test_secret_key")

    def test_rbac_initialization(self):
        """Test RBAC manager initialization."""
        assert self.rbac.secret_key == "test_secret_key"
        assert len(self.rbac._users) == 0
        assert len(self.rbac._api_keys) == 0

    def test_create_user(self):
        """Test user creation."""
        user = self.rbac.create_user("testuser", Role.USER, rate_limit=50)

        assert user.username == "testuser"
        assert user.role == Role.USER
        assert user.rate_limit == 50
        assert user.api_key.startswith("nb_")
        assert user.created_at is not None

        # User should be stored
        assert "testuser" in self.rbac._users
        assert self.rbac._users["testuser"] == user

    def test_duplicate_user_creation(self):
        """Test duplicate user creation fails."""
        self.rbac.create_user("duplicate", Role.USER)

        with pytest.raises(ValueError, match="already exists"):
            self.rbac.create_user("duplicate", Role.USER)

    def test_api_key_validation(self):
        """Test API key validation."""
        user = self.rbac.create_user("apikey_test", Role.USER)

        # Valid key
        valid_key_data = self.rbac.validate_api_key(user.api_key)
        assert valid_key_data is not None
        assert valid_key_data.username == "apikey_test"
        assert valid_key_data.is_active is True

        # Invalid key
        invalid_key_data = self.rbac.validate_api_key("invalid_key")
        assert invalid_key_data is None

    def test_permission_checking(self):
        """Test permission checking."""
        # Create user role
        user_key = self.rbac.create_user("user_role", Role.USER).api_key

        # User should have read and process permissions
        assert self.rbac.check_permission(user_key, Permission.READ) is True
        assert self.rbac.check_permission(user_key, Permission.PROCESS) is True
        assert self.rbac.check_permission(user_key, Permission.ADMIN) is False
        assert self.rbac.check_permission(user_key, Permission.DELETE) is False

        # Create admin role
        admin_key = self.rbac.create_user("admin_role", Role.ADMIN).api_key

        # Admin should have all permissions
        assert self.rbac.check_permission(admin_key, Permission.READ) is True
        assert self.rbac.check_permission(admin_key, Permission.PROCESS) is True
        assert self.rbac.check_permission(admin_key, Permission.ADMIN) is True
        assert self.rbac.check_permission(admin_key, Permission.DELETE) is True

    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        user = self.rbac.create_user("ratelimit_test", Role.USER, rate_limit=3)
        api_key = user.api_key

        # First 3 requests should be allowed
        assert self.rbac.check_rate_limit(api_key) is True
        assert self.rbac.check_rate_limit(api_key) is True
        assert self.rbac.check_rate_limit(api_key) is True

        # 4th request should be rate limited
        assert self.rbac.check_rate_limit(api_key) is False

        # Wait a bit and check status
        time.sleep(1.1)  # Wait for rate limit window

        # Should still be rate limited (window hasn't reset yet)
        status = self.rbac.get_rate_limit_status(api_key)
        assert status["remaining"] == 0

    def test_jwt_token_creation(self):
        """Test JWT token creation and verification."""
        user = self.rbac.create_user("jwt_test", Role.USER)

        # Create token
        token = self.rbac.create_jwt_token("jwt_test")
        assert token is not None
        assert isinstance(token, str)

        # Verify token
        payload = self.rbac.verify_jwt_token(token)
        assert payload is not None
        assert payload["sub"] == "jwt_test"
        assert payload["role"] == "USER"
        assert "permissions" in payload
        assert "exp" in payload
        assert "iat" in payload

    def test_jwt_token_expiry(self):
        """Test JWT token expiry."""
        user = self.rbac.create_user("expiry_test", Role.USER)

        # Create short-lived token
        token = self.rbac.create_jwt_token(
            "expiry_test", expires_delta=timedelta(seconds=1)
        )

        # Should be valid immediately
        payload = self.rbac.verify_jwt_token(token)
        assert payload is not None

        # Wait for expiry
        time.sleep(1.1)

        # Should be expired
        payload = self.rbac.verify_jwt_token(token)
        assert payload is None

    def test_api_key_revocation(self):
        """Test API key revocation."""
        user = self.rbac.create_user("revoke_test", Role.USER)
        api_key = user.api_key

        # Key should be valid initially
        assert self.rbac.validate_api_key(api_key) is not None

        # Revoke key
        revoked = self.rbac.revoke_api_key(api_key)
        assert revoked is True

        # Key should be invalid after revocation
        assert self.rbac.validate_api_key(api_key) is None

    def test_request_authentication(self):
        """Test request authentication."""
        user = self.rbac.create_user("auth_test", Role.USER)
        api_key = user.api_key

        # Test API key authentication
        result = self.rbac.authenticate_request(
            api_key=api_key, required_permission=Permission.PROCESS
        )

        assert result["authenticated"] is True
        assert result["authorized"] is True
        assert result["username"] == "auth_test"
        assert Permission.PROCESS.value in result["permissions"]

        # Test insufficient permissions
        result = self.rbac.authenticate_request(
            api_key=api_key, required_permission=Permission.ADMIN
        )

        assert result["authenticated"] is True
        assert result["authorized"] is False
        assert "Insufficient permissions" in result["error"]

    def test_jwt_authentication(self):
        """Test JWT authentication."""
        user = self.rbac.create_user("jwt_auth_test", Role.USER)
        token = self.rbac.create_jwt_token("jwt_auth_test")

        # Test JWT authentication
        result = self.rbac.authenticate_request(
            jwt_token=token, required_permission=Permission.PROCESS
        )

        assert result["authenticated"] is True
        assert result["authorized"] is True
        assert result["username"] == "jwt_auth_test"

        # Test invalid token
        result = self.rbac.authenticate_request(
            jwt_token="invalid.jwt.token", required_permission=Permission.PROCESS
        )

        assert result["authenticated"] is False
        assert result["error"] == "Invalid or expired token"

    def test_user_api_key_management(self):
        """Test listing user API keys."""
        user = self.rbac.create_user("multi_key_test", Role.USER)

        # Create additional API keys manually for testing
        key1 = APIKey(
            key="key1",
            username="multi_key_test",
            permissions={Permission.PROCESS},
            created_at=datetime.utcnow(),
        )
        key2 = APIKey(
            key="key2",
            username="multi_key_test",
            permissions={Permission.READ},
            created_at=datetime.utcnow(),
        )

        self.rbac._api_keys["key1"] = key1
        self.rbac._api_keys["key2"] = key2

        # List user's API keys
        user_keys = self.rbac.list_user_api_keys("multi_key_test")
        key_names = [k.key for k in user_keys]

        assert user.api_key in key_names
        assert "key1" in key_names
        assert "key2" in key_names


class TestCreateRbacSystem:
    """Test RBAC system factory function."""

    def test_create_rbac_system(self):
        """Test creating RBAC system with defaults."""
        if not JWT_AVAILABLE:
            pytest.skip("JWT dependencies not available")

        rbac = create_rbac_system(secret_key="factory_test")

        # Should have default users
        assert "admin" in rbac._users
        assert "demo" in rbac._users

        # Check roles
        assert rbac._users["admin"].role == Role.ADMIN
        assert rbac._users["demo"].role == Role.USER

        # Should have API keys
        assert rbac._users["admin"].api_key.startswith("nb_")
        assert rbac._users["demo"].api_key.startswith("nb_")


class TestUserAndAPIKeyClasses:
    """Test User and APIKey data classes."""

    def test_user_creation(self):
        """Test User class creation."""
        created_at = datetime.utcnow()
        user = User(
            username="testuser",
            role=Role.USER,
            api_key="test_key",
            created_at=created_at,
            rate_limit=200,
        )

        assert user.username == "testuser"
        assert user.role == Role.USER
        assert user.api_key == "test_key"
        assert user.created_at == created_at
        assert user.rate_limit == 200
        assert user.last_accessed is None

    def test_api_key_creation(self):
        """Test APIKey class creation."""
        created_at = datetime.utcnow()
        expires_at = created_at + timedelta(days=30)

        api_key = APIKey(
            key="test_api_key",
            username="testuser",
            permissions={Permission.READ, Permission.PROCESS},
            created_at=created_at,
            expires_at=expires_at,
            is_active=True,
        )

        assert api_key.key == "test_api_key"
        assert api_key.username == "testuser"
        assert Permission.READ in api_key.permissions
        assert Permission.PROCESS in api_key.permissions
        assert api_key.created_at == created_at
        assert api_key.expires_at == expires_at
        assert api_key.is_active is True


class TestRBACIntegration:
    """Integration tests for RBAC system."""

    def setup_method(self):
        """Setup integration test environment."""
        if not JWT_AVAILABLE:
            pytest.skip("JWT dependencies not available")

        self.rbac = RBACManager(secret_key="integration_test")

    def test_complete_auth_flow(self):
        """Test complete authentication and authorization flow."""
        # Create users with different roles
        viewer = self.rbac.create_user("viewer", Role.VIEWER, rate_limit=10)
        user = self.rbac.create_user("user", Role.USER, rate_limit=50)
        admin = self.rbac.create_user("admin", Role.ADMIN, rate_limit=1000)

        # Test permission hierarchy
        viewer_result = self.rbac.authenticate_request(
            api_key=viewer.api_key, required_permission=Permission.READ
        )
        assert viewer_result["authorized"] is True

        viewer_process_result = self.rbac.authenticate_request(
            api_key=viewer.api_key, required_permission=Permission.PROCESS
        )
        assert viewer_process_result["authorized"] is False

        # User can process intents
        user_result = self.rbac.authenticate_request(
            api_key=user.api_key, required_permission=Permission.PROCESS
        )
        assert user_result["authorized"] is True

        # Admin can do everything
        admin_result = self.rbac.authenticate_request(
            api_key=admin.api_key, required_permission=Permission.DELETE
        )
        assert admin_result["authorized"] is True

    def test_rate_limit_per_user(self):
        """Test that rate limits are enforced per user."""
        # Create user with low rate limit
        user = self.rbac.create_user("lowlimit", Role.USER, rate_limit=2)
        api_key = user.api_key

        # First two requests should succeed
        success_count = 0
        for i in range(2):
            result = self.rbac.authenticate_request(
                api_key=api_key, required_permission=Permission.READ
            )
            if result["authenticated"] and result["authorized"]:
                success_count += 1

        assert success_count == 2

        # Third request should be rate limited
        result = self.rbac.authenticate_request(
            api_key=api_key, required_permission=Permission.READ
        )
        assert "Rate limit exceeded" in result["error"]

    def test_jwt_and_api_key_interoperability(self):
        """Test that both JWT and API keys work."""
        user = self.rbac.create_user("interop_test", Role.USER)

        # Test API key auth
        api_result = self.rbac.authenticate_request(
            api_key=user.api_key, required_permission=Permission.PROCESS
        )
        assert api_result["authenticated"] is True

        # Test JWT auth
        token = self.rbac.create_jwt_token("interop_test")
        jwt_result = self.rbac.authenticate_request(
            jwt_token=token, required_permission=Permission.PROCESS
        )
        assert jwt_result["authenticated"] is True

        # Both should provide same username
        assert api_result["username"] == jwt_result["username"]


# Skip all tests if JWT not available
pytestmark = pytest.mark.skipif(
    not JWT_AVAILABLE,
    reason="JWT dependencies not available. Install with: pip install python-jose[cryptography] passlib[bcrypt]",
)
