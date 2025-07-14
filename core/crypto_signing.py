"""
Cryptographic Signing System
Provides secure signing and verification of actions
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

logger = logging.getLogger(__name__)

# Key file paths
SIGNING_KEY_PATH = "keys/signing_key.pem"
VERIFICATION_KEY_PATH = "keys/verification_key.pem"

# Ensure keys directory exists
os.makedirs("keys", exist_ok=True)


class ActionSigner:
    """
    Signs and verifies actions using RSA-PSS
    Ensures action integrity and non-repudiation
    """

    def __init__(self):
        self._load_or_generate_keys()

    def _load_or_generate_keys(self):
        """Load existing keys or generate new ones"""
        try:
            # Try to load existing keys
            with open(SIGNING_KEY_PATH, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
                )
            with open(VERIFICATION_KEY_PATH, "rb") as f:
                self.public_key = serialization.load_pem_public_key(
                    f.read(), backend=default_backend()
                )
        except Exception:
            # Generate new key pair
            self.private_key = rsa.generate_private_key(
                public_exponent=65537, key_size=2048, backend=default_backend()
            )
            self.public_key = self.private_key.public_key()

            # Save keys securely
            self._save_keys()

    def _save_keys(self):
        """Save keys with appropriate permissions"""
        # Save private key (readable only by owner)
        private_pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        with open(SIGNING_KEY_PATH, "wb") as f:
            f.write(private_pem)
        os.chmod(SIGNING_KEY_PATH, 0o600)  # Owner read/write only

        # Save public key (readable by all)
        public_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        with open(VERIFICATION_KEY_PATH, "wb") as f:
            f.write(public_pem)
        os.chmod(VERIFICATION_KEY_PATH, 0o644)  # All can read

    def sign_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sign an approved action
        Returns the action with signature added
        """
        # Create canonical representation
        canonical = self._canonicalize_action(action)

        # Create signature
        signature = self.private_key.sign(
            canonical.encode("utf-8"),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )

        # Add signature to action
        signed_action = action.copy()
        signed_action["_signature"] = {
            "signature": base64.b64encode(signature).decode("utf-8"),
            "signed_at": datetime.now().isoformat(),
            "algorithm": "RSA-PSS-SHA256",
        }

        return signed_action

    def verify_signature(self, signed_action: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Verify a signed action
        Returns (is_valid, reason)
        """
        if "_signature" not in signed_action:
            return False, "No signature present"

        signature_data = signed_action["_signature"]

        # Extract signature
        try:
            signature = base64.b64decode(signature_data["signature"])
        except Exception:
            return False, "Invalid signature format"

        # Remove signature for verification
        action = {k: v for k, v in list(signed_action.items()) if k != "_signature"}
        canonical = self._canonicalize_action(action)

        # Verify signature
        try:
            self.public_key.verify(
                signature,
                canonical.encode("utf-8"),
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )
            return True, "Valid signature"
        except InvalidSignature:
            return False, "Invalid signature"
        except Exception as e:
            return False, f"Verification error: {e}"

    def _canonicalize_action(self, action: Dict[str, Any]) -> str:
        """Create canonical string representation of action"""
        # Sort keys and create deterministic JSON
        return json.dumps(action, sort_keys=True, separators=(",", ":"))

    def get_fingerprint(self) -> str:
        """Get fingerprint of the public key"""
        public_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return hashlib.sha256(public_bytes).hexdigest()[:16]


class ApprovalToken:
    """
    Time-limited approval tokens for actions
    """

    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or os.urandom(32)

    def generate_token(self, action: str, user: str, expires_in: int = 300) -> str:
        """
        Generate time-limited approval token
        expires_in: seconds until expiration
        """
        payload = {
            "action": action,
            "user": user,
            "issued_at": time.time(),
            "expires_at": time.time() + expires_in
        }

        # Create token
        message = json.dumps(payload, sort_keys=True)
        signature = hmac.new(self.secret_key, message.encode("utf-8"), hashlib.sha256).hexdigest()

        # Combine payload and signature
        token = base64.b64encode(f"{message}|{signature}".encode("utf-8")).decode("utf-8")

        return token

    def verify_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verify and decode approval token
        Returns (is_valid, payload)
        """
        try:
            # Decode token
            decoded = base64.b64decode(token).decode("utf-8")
            message, signature = decoded.split("|")

            # Verify signature
            expected_signature = hmac.new(self.secret_key, message.encode("utf-8"), hashlib.sha256).hexdigest()

            if not hmac.compare_digest(signature, expected_signature):
                return False, None

            # Decode payload
            payload = json.loads(message)

            # Check expiration
            if time.time() > payload["expires_at"]:
                return False, None

            return True, payload

        except Exception:
            return False, None


class SecureActionApproval:
    """
    Combines signing and tokens for secure action approval
    """

    def __init__(self):
        self.signer = ActionSigner()
        self.token_generator = ApprovalToken()
        self.pending_approvals: Dict[str, Dict[str, Any]] = {}

    def request_approval(self, action: Dict[str, Any], requested_by: str) -> str:
        """
        Request approval for an action
        Returns approval request ID
        """
        request_id = hashlib.sha256(f"{json.dumps(action)}{time.time()}".encode()).hexdigest()[:16]

        self.pending_approvals[request_id] = {
            "action": action,
            "requested_by": requested_by,
            "requested_at": datetime.now().isoformat(),
            "status": "pending",
        }

        return request_id

    def approve_action(self, request_id: str, approved_by: str) -> Optional[Dict[str, Any]]:
        """
        Approve a pending action
        Returns signed action with approval token
        """
        if request_id not in self.pending_approvals:
            return None

        approval = self.pending_approvals[request_id]
        if approval["status"] != "pending":
            return None

        # Generate approval token
        token = self.token_generator.generate_token(
            action=request_id, user=approved_by, expires_in=300  # 5 minutes
        )

        # Sign the action
        action = approval["action"].copy()
        action["_approval"] = {
            "approved_by": approved_by,
            "approved_at": datetime.now().isoformat(),
            "token": token,
        }

        signed_action = self.signer.sign_action(action)

        # Update approval status
        approval["status"] = "approved"
        approval["approved_by"] = approved_by
        approval["approved_at"] = datetime.now().isoformat()

        return signed_action

    def verify_approved_action(self, signed_action: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Verify an approved action
        Returns (is_valid, reason)
        """
        # Verify signature
        signature_valid, signature_reason = self.signer.verify_signature(signed_action)
        if not signature_valid:
            return False, f"Signature verification failed: {signature_reason}"

        # Verify approval token
        if "_approval" not in signed_action:
            return False, "No approval present"

        approval = signed_action["_approval"]
        if "token" not in approval:
            return False, "No approval token present"

        token_valid, payload = self.token_generator.verify_token(approval["token"])
        if not token_valid:
            return False, "Approval token is invalid or expired"

        return True, "Action is valid and approved"


# Global instance
_secure_approval = None


def get_secure_approval() -> SecureActionApproval:
    """Get global secure approval instance"""
    global _secure_approval
    if _secure_approval is None:
        _secure_approval = SecureActionApproval()
    return _secure_approval


# Convenience functions
def sign_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """Sign an action"""
    signer = ActionSigner()
    return signer.sign_action(action)


def verify_action_signature(signed_action: Dict[str, Any]) -> Tuple[bool, str]:
    """Verify an action signature"""
    signer = ActionSigner()
    return signer.verify_signature(signed_action)
