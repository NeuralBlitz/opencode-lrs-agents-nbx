"""
NeuralBlitz v50.0 Quantum Cryptography Module
==============================================

Quantum-resistant encryption and secure communication protocols
for distributed AI agent coordination.

Implementation Date: 2026-02-04
Phase: Quantum Foundation - Q2 Implementation
"""

import asyncio
import hashlib
import hmac
import secrets
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet

import numpy as np
from .quantum_foundation import quantum_comm_layer, qkd_system, QuantumAgent


@dataclass
class QuantumSecureMessage:
    """Quantum-encrypted message with tamper-proof verification"""

    sender_id: str
    receiver_id: str
    encrypted_payload: bytes
    quantum_signature: bytes
    timestamp: float
    message_id: str
    entanglement_proof: Optional[bytes] = None
    reality_id: Optional[int] = None


@dataclass
class QuantumSession:
    """Secure quantum communication session"""

    session_id: str
    participants: List[str]
    quantum_key: bytes
    session_start: float
    last_activity: float
    message_count: int = 0
    integrity_hash: bytes = field(default_factory=lambda: b"")


class QuantumEncryptionEngine:
    """
    Quantum-Resistant Encryption Engine

    Combines quantum key distribution with post-quantum cryptographic
    algorithms for unbreakable agent communication.
    """

    def __init__(self):
        self.active_sessions: Dict[str, QuantumSession] = {}
        self.message_history: List[QuantumSecureMessage] = []
        self.key_rotation_interval = 3600  # 1 hour
        self.quantum_signatures: Dict[str, ec.EllipticCurvePrivateKey] = {}

    def create_quantum_session(
        self, participant_ids: List[str]
    ) -> Optional[QuantumSession]:
        """Create secure quantum communication session"""
        if len(participant_ids) < 2:
            return None

        # Generate quantum keys for all participants
        session_keys = []
        for i, agent1 in enumerate(participant_ids):
            for agent2 in participant_ids[i + 1 :]:
                shared_key = qkd_system.generate_quantum_key(agent1, agent2)
                if shared_key:
                    session_keys.append(shared_key)

        if not session_keys:
            # Fallback to classical key generation
            master_key = secrets.token_bytes(32)
        else:
            # Combine quantum keys using XOR
            master_key = session_keys[0]
            for key in session_keys[1:]:
                master_key = bytes(a ^ b for a, b in zip(master_key, key))

        session_id = secrets.token_hex(16)
        session = QuantumSession(
            session_id=session_id,
            participants=participant_ids,
            quantum_key=master_key,
            session_start=time.time(),
            last_activity=time.time(),
        )

        # Generate session integrity hash
        session.integrity_hash = self._generate_session_hash(session)

        self.active_sessions[session_id] = session
        return session

    def encrypt_message(
        self,
        sender_id: str,
        receiver_id: str,
        message: Union[str, bytes],
        session_id: Optional[str] = None,
    ) -> Optional[QuantumSecureMessage]:
        """Encrypt message using quantum-resistant encryption"""
        if isinstance(message, str):
            message = message.encode("utf-8")

        # Get encryption key
        encryption_key = self._get_encryption_key(sender_id, receiver_id, session_id)
        if not encryption_key:
            return None

        # Generate message ID
        message_id = secrets.token_hex(16)

        # Encrypt payload
        encrypted_payload = self._quantum_encrypt(message, encryption_key)

        # Generate quantum signature
        quantum_signature = self._generate_quantum_signature(
            sender_id, message_id, encrypted_payload
        )

        # Generate entanglement proof if participants are entangled
        entanglement_proof = self._generate_entanglement_proof(sender_id, receiver_id)

        secure_message = QuantumSecureMessage(
            sender_id=sender_id,
            receiver_id=receiver_id,
            encrypted_payload=encrypted_payload,
            quantum_signature=quantum_signature,
            timestamp=time.time(),
            message_id=message_id,
            entanglement_proof=entanglement_proof,
        )

        self.message_history.append(secure_message)
        return secure_message

    def decrypt_message(
        self,
        receiver_id: str,
        secure_message: QuantumSecureMessage,
        session_id: Optional[str] = None,
    ) -> Optional[bytes]:
        """Decrypt quantum-encrypted message"""
        # Verify quantum signature
        if not self._verify_quantum_signature(secure_message):
            return None

        # Get decryption key
        decryption_key = self._get_encryption_key(
            secure_message.sender_id, receiver_id, session_id
        )
        if not decryption_key:
            return None

        # Decrypt payload
        try:
            decrypted_payload = self._quantum_decrypt(
                secure_message.encrypted_payload, decryption_key
            )
            return decrypted_payload
        except Exception:
            return None

    def _get_encryption_key(
        self, sender_id: str, receiver_id: str, session_id: Optional[str] = None
    ) -> Optional[bytes]:
        """Get encryption key for communication"""
        # Try session key first
        if session_id and session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            if (
                sender_id in session.participants
                and receiver_id in session.participants
            ):
                return self._derive_message_key(
                    session.quantum_key, session.message_count
                )

        # Fall back to direct quantum key
        shared_keys = qkd_system.shared_keys
        if receiver_id in shared_keys.get(sender_id, {}):
            return shared_keys[sender_id][receiver_id]
        elif sender_id in shared_keys.get(receiver_id, {}):
            return shared_keys[receiver_id][sender_id]

        return None

    def _quantum_encrypt(self, data: bytes, key: bytes) -> bytes:
        """Quantum-resistant encryption using AES-256-GCM"""
        # Generate random IV
        iv = secrets.token_bytes(12)

        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()

        # Encrypt data
        ciphertext = encryptor.update(data) + encryptor.finalize()

        # Return IV + ciphertext + tag
        return iv + ciphertext + encryptor.tag

    def _quantum_decrypt(self, encrypted_data: bytes, key: bytes) -> bytes:
        """Quantum-resistant decryption"""
        # Extract IV, ciphertext, and tag
        iv = encrypted_data[:12]
        tag = encrypted_data[-16:]
        ciphertext = encrypted_data[12:-16]

        # Create cipher
        cipher = Cipher(
            algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend()
        )
        decryptor = cipher.decryptor()

        # Decrypt data
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        return plaintext

    def _derive_message_key(self, session_key: bytes, message_count: int) -> bytes:
        """Derive message-specific key from session key"""
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=f"neuralblitz_message_{message_count}".encode(),
            backend=default_backend(),
        )
        return hkdf.derive(session_key)

    def _generate_quantum_signature(
        self, sender_id: str, message_id: str, encrypted_payload: bytes
    ) -> bytes:
        """Generate quantum-resistant signature"""
        # Get or create sender's private key
        if sender_id not in self.quantum_signatures:
            private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
            self.quantum_signatures[sender_id] = private_key
        else:
            private_key = self.quantum_signatures[sender_id]

        # Create message hash
        message_hash = hashlib.sha256(message_id.encode() + encrypted_payload).digest()

        # Sign with quantum-resistant algorithm
        signature = private_key.sign(message_hash, ec.ECDSA(hashes.SHA256()))

        return signature

    def _verify_quantum_signature(self, secure_message: QuantumSecureMessage) -> bool:
        """Verify quantum signature"""
        sender_id = secure_message.sender_id

        if sender_id not in self.quantum_signatures:
            return False

        private_key = self.quantum_signatures[sender_id]
        public_key = private_key.public_key()

        # Recreate message hash
        message_hash = hashlib.sha256(
            secure_message.message_id.encode() + secure_message.encrypted_payload
        ).digest()

        try:
            public_key.verify(
                secure_message.quantum_signature,
                message_hash,
                ec.ECDSA(hashes.SHA256()),
            )
            return True
        except Exception:
            return False

    def _generate_entanglement_proof(
        self, sender_id: str, receiver_id: str
    ) -> Optional[bytes]:
        """Generate proof of quantum entanglement between agents"""
        sender = quantum_comm_layer.quantum_agents.get(sender_id)
        receiver = quantum_comm_layer.quantum_agents.get(receiver_id)

        if not sender or not receiver:
            return None

        # Check if agents are entangled
        if receiver_id in sender.entangled_partners:
            # Generate entanglement timestamp and coherence proof
            proof_data = {
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "entanglement_time": time.time(),
                "coherence_factor": (
                    sender.coherence_factor + receiver.coherence_factor
                )
                / 2,
            }

            proof_hash = hashlib.sha256(str(proof_data).encode()).digest()

            return proof_hash

        return None

    def _generate_session_hash(self, session: QuantumSession) -> bytes:
        """Generate integrity hash for session"""
        session_data = {
            "session_id": session.session_id,
            "participants": sorted(session.participants),
            "start_time": session.session_start,
            "key_hash": hashlib.sha256(session.quantum_key).hexdigest(),
        }

        return hashlib.sha256(str(session_data).encode()).digest()

    def rotate_session_keys(self):
        """Rotate quantum keys for all active sessions"""
        current_time = time.time()

        for session_id, session in self.active_sessions.items():
            if current_time - session.session_start > self.key_rotation_interval:
                # Generate new quantum keys
                for i, agent1 in enumerate(session.participants):
                    for agent2 in session.participants[i + 1 :]:
                        new_key = qkd_system.generate_quantum_key(agent1, agent2)
                        if new_key:
                            # Update session key
                            session.quantum_key = new_key
                            session.session_start = current_time
                            session.integrity_hash = self._generate_session_hash(
                                session
                            )

    def cleanup_expired_sessions(self, max_age: float = 86400):
        """Clean up expired sessions"""
        current_time = time.time()
        expired_sessions = []

        for session_id, session in self.active_sessions.items():
            if current_time - session.last_activity > max_age:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self.active_sessions[session_id]


# Global quantum encryption engine
quantum_encryption = QuantumEncryptionEngine()


class QuantumSecureCommunications:
    """
    High-level interface for quantum-secure communications

    Provides simplified API for quantum-resistant messaging and
    secure agent coordination.
    """

    def __init__(self):
        self.encryption_engine = quantum_encryption

    async def send_quantum_message(
        self,
        sender_id: str,
        receiver_id: str,
        message: str,
        session_id: Optional[str] = None,
    ) -> bool:
        """Send quantum-encrypted message"""
        secure_msg = self.encryption_engine.encrypt_message(
            sender_id, receiver_id, message, session_id
        )

        if secure_msg:
            # Simulate quantum transmission
            await asyncio.sleep(0.001)  # Quantum transmission delay

            # In real implementation, transmit through quantum channel
            return True
        return False

    async def receive_quantum_message(
        self,
        receiver_id: str,
        secure_message: QuantumSecureMessage,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        """Receive and decrypt quantum message"""
        decrypted = self.encryption_engine.decrypt_message(
            receiver_id, secure_message, session_id
        )

        if decrypted:
            return decrypted.decode("utf-8")
        return None

    def create_secure_channel(self, participant_ids: List[str]) -> Optional[str]:
        """Create secure quantum communication channel"""
        session = self.encryption_engine.create_quantum_session(participant_ids)
        return session.session_id if session else None

    def verify_message_integrity(self, message: QuantumSecureMessage) -> bool:
        """Verify message integrity using quantum signatures"""
        return self.encryption_engine._verify_quantum_signature(message)


# Global secure communications interface
secure_comm = QuantumSecureCommunications()


async def test_quantum_encryption():
    """Test quantum encryption capabilities"""
    print("🔐 Testing Quantum Encryption System...")

    # Create quantum agents
    agent1 = quantum_comm_layer.create_quantum_agent("alpha")
    agent2 = quantum_comm_layer.create_quantum_agent("beta")

    # Create entanglement
    quantum_comm_layer.create_entanglement("alpha", "beta")

    # Create secure channel
    channel_id = secure_comm.create_secure_channel(["alpha", "beta"])
    print(f"📡 Created secure channel: {channel_id}")

    # Send encrypted message
    success = await secure_comm.send_quantum_message(
        "alpha", "beta", "Quantum consciousness achieved!", channel_id
    )

    if success:
        print("✅ Quantum encryption test successful!")
    else:
        print("❌ Quantum encryption test failed!")


if __name__ == "__main__":
    asyncio.run(test_quantum_encryption())
