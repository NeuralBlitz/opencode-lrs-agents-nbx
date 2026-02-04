"""
NeuralBlitz V50 - Service Bus
Central message broker with bidirectional streaming.
"""

import asyncio
from typing import Dict, List, Set, Optional, Callable, Any, Coroutine
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import threading

from .protocol import (
    ComponentType,
    MessageType,
    Message,
    Priority,
    ComponentInfo,
    ComponentAdapter,
)

logger = logging.getLogger("NeuralBlitz.ServiceBus")


class ServiceRegistry:
    """
    Registry for all ecosystem components.

    Tracks available services, their health, and routing information.
    """

    def __init__(self):
        self._services: Dict[ComponentType, List[ComponentInfo]] = defaultdict(list)
        self._instance_map: Dict[str, ComponentInfo] = {}
        self._health_status: Dict[str, str] = {}
        self._last_heartbeat: Dict[str, datetime] = {}
        self._lock = threading.RLock()

    def register(self, info: ComponentInfo):
        """Register a component."""
        with self._lock:
            self._services[info.component_type].append(info)
            self._instance_map[info.instance_id] = info
            self._health_status[info.instance_id] = "healthy"
            self._last_heartbeat[info.instance_id] = datetime.utcnow()
            logger.info(f"Registered {info.component_type.value}: {info.instance_id}")

    def unregister(self, instance_id: str):
        """Unregister a component."""
        with self._lock:
            if instance_id in self._instance_map:
                info = self._instance_map[instance_id]
                self._services[info.component_type] = [
                    s
                    for s in self._services[info.component_type]
                    if s.instance_id != instance_id
                ]
                del self._instance_map[instance_id]
                del self._health_status[instance_id]
                del self._last_heartbeat[instance_id]
                logger.info(f"Unregistered {instance_id}")

    def get_service(self, component_type: ComponentType) -> Optional[ComponentInfo]:
        """Get a healthy service of given type (round-robin)."""
        with self._lock:
            services = self._services.get(component_type, [])
            healthy = [
                s
                for s in services
                if self._health_status.get(s.instance_id) == "healthy"
            ]

            if healthy:
                # Simple round-robin: pick first and rotate
                return healthy[0]
            return None

    def get_all_services(self, component_type: ComponentType) -> List[ComponentInfo]:
        """Get all services of given type."""
        with self._lock:
            return list(self._services.get(component_type, []))

    def get_by_instance_id(self, instance_id: str) -> Optional[ComponentInfo]:
        """Get service by instance ID."""
        with self._lock:
            return self._instance_map.get(instance_id)

    def update_health(self, instance_id: str, status: str):
        """Update health status."""
        with self._lock:
            if instance_id in self._health_status:
                self._health_status[instance_id] = status
                self._last_heartbeat[instance_id] = datetime.utcnow()

    def check_stale_services(self, timeout_seconds: int = 60) -> List[str]:
        """Find services that haven't sent heartbeat."""
        with self._lock:
            stale = []
            cutoff = datetime.utcnow() - timedelta(seconds=timeout_seconds)

            for instance_id, last_seen in self._last_heartbeat.items():
                if last_seen < cutoff:
                    stale.append(instance_id)
                    self._health_status[instance_id] = "stale"

            return stale

    def discover_by_capability(self, capability: str) -> List[ComponentInfo]:
        """Discover services with specific capability."""
        with self._lock:
            matching = []
            for services in self._services.values():
                for service in services:
                    if capability in service.capabilities:
                        matching.append(service)
            return matching


class SubscriptionManager:
    """
    Manages pub/sub subscriptions for event distribution.
    """

    def __init__(self):
        self._subscriptions: Dict[str, Set[str]] = defaultdict(
            set
        )  # event_type -> instance_ids
        self._filters: Dict[str, Dict[str, Any]] = {}  # instance_id -> filter
        self._lock = threading.RLock()

    def subscribe(
        self,
        instance_id: str,
        event_types: List[str],
        filters: Optional[Dict[str, Any]] = None,
    ):
        """Subscribe to event types."""
        with self._lock:
            for event_type in event_types:
                self._subscriptions[event_type].add(instance_id)

            if filters:
                self._filters[instance_id] = filters

            logger.info(f"{instance_id} subscribed to {event_types}")

    def unsubscribe(self, instance_id: str, event_types: Optional[List[str]] = None):
        """Unsubscribe from events."""
        with self._lock:
            if event_types is None:
                # Unsubscribe from all
                for subscribers in self._subscriptions.values():
                    subscribers.discard(instance_id)
                self._filters.pop(instance_id, None)
            else:
                for event_type in event_types:
                    self._subscriptions[event_type].discard(instance_id)

    def get_subscribers(
        self, event_type: str, payload: Optional[Dict] = None
    ) -> List[str]:
        """Get all subscribers for an event type, filtered if needed."""
        with self._lock:
            subscribers = list(self._subscriptions.get(event_type, set()))

            if payload:
                # Apply filters
                filtered = []
                for sub_id in subscribers:
                    filter_config = self._filters.get(sub_id, {})
                    if self._matches_filter(payload, filter_config):
                        filtered.append(sub_id)
                return filtered

            return subscribers

    def _matches_filter(self, payload: Dict, filter_config: Dict) -> bool:
        """Check if payload matches subscription filter."""
        for key, value in filter_config.items():
            if key in payload and payload[key] != value:
                return False
        return True


class ServiceBus:
    """
    Central service bus for bidirectional communication.

    Features:
    - Message routing
    - Bidirectional streaming
    - Service discovery
    - Pub/sub events
    - Load balancing
    - Health monitoring
    """

    def __init__(self):
        self.registry = ServiceRegistry()
        self.subscriptions = SubscriptionManager()
        self._adapters: Dict[str, ComponentAdapter] = {}
        self._response_futures: Dict[str, asyncio.Future] = {}
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()

    def register_adapter(self, adapter: ComponentAdapter):
        """Register a component adapter."""
        with self._lock:
            adapter.bus = self
            self._adapters[adapter.instance_id] = adapter
            self.registry.register(adapter.get_info())
            logger.info(f"Registered adapter: {adapter.instance_id}")

    def unregister_adapter(self, instance_id: str):
        """Unregister a component adapter."""
        with self._lock:
            if instance_id in self._adapters:
                del self._adapters[instance_id]
                self.registry.unregister(instance_id)
                self.subscriptions.unsubscribe(instance_id)

    async def send(self, message: Message, timeout: float = 30.0) -> Optional[Message]:
        """
        Send a message and optionally wait for response.

        Args:
            message: Message to send
            timeout: Seconds to wait for response

        Returns:
            Response message if target responds, None otherwise
        """
        # Add to trace
        message.trace.append(f"bus:{datetime.utcnow().isoformat()}")

        # Route to target
        if message.target:
            target_adapter = self._adapters.get(self._get_instance_id(message.target))

            if target_adapter:
                try:
                    response = await target_adapter.handle_message(message)
                    return response
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    return self._create_error_response(message, str(e))
            else:
                logger.warning(f"No adapter for target: {message.target}")
                return None
        else:
            # Broadcast
            await self._broadcast(message)
            return None

    async def send_and_wait(
        self, message: Message, timeout: float = 30.0
    ) -> Optional[Message]:
        """Send message and wait for response."""
        # Create future for response
        future = asyncio.Future()
        self._response_futures[message.id] = future

        # Send
        await self.send(message)

        # Wait for response
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for response to {message.id}")
            return None
        finally:
            self._response_futures.pop(message.id, None)

    async def _broadcast(self, message: Message):
        """Broadcast message to all relevant adapters."""
        tasks = []

        for adapter in self._adapters.values():
            # Don't send back to source
            if adapter.component_type != message.source:
                task = asyncio.create_task(adapter.handle_message(message))
                tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _get_instance_id(self, component_type: ComponentType) -> Optional[str]:
        """Get instance ID for component type."""
        info = self.registry.get_service(component_type)
        return info.instance_id if info else None

    def _create_error_response(self, original: Message, error: str) -> Message:
        """Create error response message."""
        return Message(
            source=ComponentType.GATEWAY,
            target=original.source,
            reply_to=original.id,
            msg_type=MessageType.ERROR,
            payload={"error": error, "original_message_id": original.id},
            correlation_id=original.correlation_id,
        )

    async def publish_event(self, event_type: str, payload: Dict[str, Any]):
        """Publish event to all subscribers."""
        subscribers = self.subscriptions.get_subscribers(event_type, payload)

        message = Message(
            source=ComponentType.GATEWAY,
            target=None,
            msg_type=MessageType.UPDATE,
            payload={
                "event_type": event_type,
                "data": payload,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        for sub_id in subscribers:
            adapter = self._adapters.get(sub_id)
            if adapter:
                asyncio.create_task(adapter.handle_message(message))

    async def start(self):
        """Start the service bus."""
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Service bus started")

    async def stop(self):
        """Stop the service bus."""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("Service bus stopped")

    async def _health_check_loop(self):
        """Background health check."""
        while self._running:
            try:
                # Check for stale services
                stale = self.registry.check_stale_services(timeout_seconds=60)
                if stale:
                    logger.warning(f"Stale services detected: {stale}")
                    for instance_id in stale:
                        self.unregister_adapter(instance_id)

                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)

    def get_stats(self) -> Dict[str, Any]:
        """Get bus statistics."""
        return {
            "registered_adapters": len(self._adapters),
            "service_types": len(self.registry._services),
            "total_services": sum(len(s) for s in self.registry._services.values()),
            "pending_responses": len(self._response_futures),
            "subscriptions": {
                event_type: len(subs)
                for event_type, subs in self.subscriptions._subscriptions.items()
            },
        }


class BidirectionalStream:
    """
    Bidirectional streaming connection between two components.

    Enables real-time data flow in both directions.
    """

    def __init__(
        self,
        stream_id: str,
        source: ComponentAdapter,
        target: ComponentAdapter,
        bus: ServiceBus,
    ):
        self.stream_id = stream_id
        self.source = source
        self.target = target
        self.bus = bus
        self._active = False
        self._queues: Dict[str, asyncio.Queue] = {
            "source_to_target": asyncio.Queue(),
            "target_to_source": asyncio.Queue(),
        }
        self._tasks: List[asyncio.Task] = []

    async def start(self):
        """Start bidirectional streaming."""
        self._active = True

        # Start consumer tasks
        self._tasks.append(asyncio.create_task(self._consume_source()))
        self._tasks.append(asyncio.create_task(self._consume_target()))

        logger.info(
            f"Stream {self.stream_id} started between {self.source.instance_id} and {self.target.instance_id}"
        )

    async def stop(self):
        """Stop streaming."""
        self._active = False

        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info(f"Stream {self.stream_id} stopped")

    async def _consume_source(self):
        """Consume messages from source."""
        while self._active:
            try:
                message = await self._queues["source_to_target"].get()
                await self.target.handle_message(message)
            except Exception as e:
                logger.error(f"Stream {self.stream_id} source error: {e}")

    async def _consume_target(self):
        """Consume messages from target."""
        while self._active:
            try:
                message = await self._queues["target_to_source"].get()
                await self.source.handle_message(message)
            except Exception as e:
                logger.error(f"Stream {self.stream_id} target error: {e}")

    async def send_from_source(self, message: Message):
        """Send message from source to target."""
        await self._queues["source_to_target"].put(message)

    async def send_from_target(self, message: Message):
        """Send message from target to source."""
        await self._queues["target_to_source"].put(message)


__all__ = [
    "ServiceBus",
    "ServiceRegistry",
    "SubscriptionManager",
    "BidirectionalStream",
]
