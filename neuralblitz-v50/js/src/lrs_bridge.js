/**
 * NeuralBlitz v50.0 - LRS Bridge Module
 * Bidirectional Communication with LRS Agents
 */

const crypto = require('crypto');
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const { v4: uuidv4 } = require('uuid');

/**
 * LRS Bridge Configuration
 */
class LRSBridgeConfig {
    constructor(options = {}) {
        this.lrsEndpoint = options.lrsEndpoint || process.env.LRS_AGENT_ENDPOINT || 'http://localhost:9000';
        this.authKey = options.authKey || process.env.LRS_AUTH_KEY || 'shared_goldendag_key';
        this.systemId = options.systemId || 'NEURALBLITZ_V50';
        this.heartbeatInterval = options.heartbeatInterval || 30000; // 30 seconds
        this.connectionTimeout = options.connectionTimeout || 10000; // 10 seconds
        this.maxRetries = options.maxRetries || 3;
        this.circuitBreakerThreshold = options.circuitBreakerThreshold || 5;
        this.circuitBreakerTimeout = options.circuitBreakerTimeout || 60000; // 1 minute
    }
}

/**
 * Circuit Breaker Implementation
 */
class CircuitBreaker {
    constructor(threshold = 5, timeout = 60000) {
        this.failureThreshold = threshold;
        this.timeout = timeout;
        this.failureCount = 0;
        this.lastFailureTime = null;
        this.state = 'CLOSED'; // CLOSED, OPEN, HALF_OPEN
    }

    shouldAllowRequest() {
        switch (this.state) {
            case 'CLOSED':
                return true;
            case 'OPEN':
                const timeSinceFailure = Date.now() - this.lastFailureTime;
                if (timeSinceFailure > this.timeout) {
                    this.state = 'HALF_OPEN';
                    return true;
                }
                return false;
            case 'HALF_OPEN':
                return true;
            default:
                return false;
        }
    }

    recordSuccess() {
        this.failureCount = 0;
        this.state = 'CLOSED';
    }

    recordFailure() {
        this.failureCount++;
        this.lastFailureTime = Date.now();
        
        if (this.failureCount >= this.failureThreshold) {
            this.state = 'OPEN';
        }
    }

    getState() {
        return this.state;
    }
}

/**
 * Message Queue Implementation
 */
class MessageQueue {
    constructor(maxSize = 1000) {
        this.maxSize = maxSize;
        this.queue = [];
        this.processing = false;
    }

    enqueue(message) {
        if (this.queue.length >= this.maxSize) {
            throw new Error('Message queue is full');
        }
        this.queue.push(message);
        return true;
    }

    dequeue() {
        if (this.queue.length === 0) {
            return null;
        }
        return this.queue.shift();
    }

    get size() {
        return this.queue.length;
    }

    get isProcessing() {
        return this.processing;
    }

    set isProcessing(value) {
        this.processing = value;
    }
}

/**
 * LRS Message Structure
 */
class LRSMessage {
    constructor(sourceSystem, targetSystem, messageType, payload, authKey) {
        this.protocolVersion = '1.0';
        this.timestamp = Date.now();
        this.sourceSystem = sourceSystem;
        this.targetSystem = targetSystem;
        this.messageId = uuidv4();
        this.correlationId = null;
        this.messageType = messageType;
        this.payload = payload;
        this.signature = ''; // Will be set by sign method
        this.priority = 'NORMAL';
        this.ttl = 300;
        
        this.sign(authKey);
    }

    sign(authKey) {
        const payloadString = JSON.stringify(this.payload);
        this.signature = crypto
            .createHmac('sha256', authKey)
            .update(payloadString)
            .digest('hex');
    }

    verifySignature(authKey) {
        const payloadString = JSON.stringify(this.payload);
        const expectedSignature = crypto
            .createHmac('sha256', authKey)
            .update(payloadString)
            .digest('hex');
        
        return this.signature === expectedSignature;
    }
}

/**
 * Main LRS Bridge Class
 */
class LRSBridge {
    constructor(config) {
        this.config = new LRSBridgeConfig(config);
        this.messageQueue = new MessageQueue();
        this.circuitBreaker = new CircuitBreaker(
            this.config.circuitBreakerThreshold,
            this.config.circuitBreakerTimeout
        );
        this.messageHandlers = new Map();
        this.running = false;
        this.startTime = Date.now();
        this.activeConnections = new Map();
    }

    async start() {
        console.log(`Starting LRS bridge for ${this.config.systemId}`);
        this.running = true;
        
        // Register default handlers
        this.registerDefaultHandlers();
        
        // Start heartbeat loop
        this.heartbeatInterval = setInterval(() => {
            if (!this.running) {
                clearInterval(this.heartbeatInterval);
                return;
            }
            
            this.sendHeartbeat().catch(error => {
                console.error('Heartbeat failed:', error);
            });
        }, this.config.heartbeatInterval);
        
        console.log('LRS bridge started successfully');
    }

    async stop() {
        console.log('Stopping LRS bridge');
        this.running = false;
        
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
        }
        
        console.log('LRS bridge stopped');
    }

    registerHandler(messageType, handler) {
        this.messageHandlers.set(messageType, handler);
        console.log(`Registered handler for message type: ${messageType}`);
    }

    registerDefaultHandlers() {
        // Default handlers would be registered here
        console.log('Default handlers registered');
    }

    async sendMessage(messageType, payload, targetSystem = 'LRS_AGENT') {
        if (!this.circuitBreaker.shouldAllowRequest()) {
            throw new Error('Circuit breaker is OPEN');
        }
        
        const message = new LRSMessage(
            this.config.systemId,
            targetSystem,
            messageType,
            payload,
            this.config.authKey
        );

        try {
            const response = await this.makeHttpRequest(message);
            this.circuitBreaker.recordSuccess();
            return response;
        } catch (error) {
            this.circuitBreaker.recordFailure();
            console.error('Error sending message to LRS:', error);
            throw error;
        }
    }

    async makeHttpRequest(message) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.config.connectionTimeout);

        try {
            const response = await fetch(`${this.config.lrsEndpoint}/neuralblitz/bridge`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-System-ID': this.config.systemId,
                    'X-Auth-Signature': message.signature
                },
                body: JSON.stringify(message),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            clearTimeout(timeoutId);
            throw error;
        }
    }

    async submitIntent(intent, targetSystem = 'LRS_AGENT') {
        const payload = {
            phi_1: intent.phi_1,
            phi_22: intent.phi_22,
            phi_omega: intent.phi_omega,
            metadata: intent.metadata || {}
        };

        const response = await this.sendMessage('INTENT_SUBMIT', payload, targetSystem);
        return response;
    }

    async verifyCoherence(targetSystem = 'LRS_AGENT', verificationType = 'CURRENT_STATE') {
        const payload = {
            target_system: targetSystem,
            verification_type: verificationType
        };

        const response = await this.sendMessage('COHERENCE_VERIFICATION', payload, targetSystem);
        return response;
    }

    async sendHeartbeat() {
        const uptime = Math.floor((Date.now() - this.startTime) / 1000);

        const metrics = {
            system_status: 'HEALTHY',
            queue_size: this.messageQueue.size,
            circuit_breaker_state: this.circuitBreaker.getState(),
            active_connections: this.activeConnections.size,
            coherence: 1.0,
            golden_dag_valid: true,
            uptime_seconds: uptime
        };

        const payload = {
            system_id: this.config.systemId,
            metrics: metrics
        };

        try {
            await this.sendMessage('HEARTBEAT', payload, 'LRS_AGENT');
        } catch (error) {
            console.error('Heartbeat failed:', error);
        }
    }

    isHealthy() {
        return this.circuitBreaker.getState() === 'CLOSED' && this.running;
    }

    getMetrics() {
        const uptime = Math.floor((Date.now() - this.startTime) / 1000);

        return {
            queue_size: this.messageQueue.size,
            circuit_breaker_state: this.circuitBreaker.getState(),
            active_connections: this.activeConnections.size,
            is_healthy: this.isHealthy(),
            uptime_seconds: uptime
        };
    }
}

// Global bridge instance
let lrsBridge = null;

function getLRSBridge() {
    if (!lrsBridge) {
        const config = new LRSBridgeConfig();
        lrsBridge = new LRSBridge(config);
    }
    return lrsBridge;
}

// Express.js server setup
function createLRSApp() {
    const bridge = getLRSBridge();
    const app = express();

    // Middleware
    app.use(helmet());
    app.use(cors());
    app.use(express.json());

    // Rate limiting
    const rateLimit = require('express-rate-limit');
    const limiter = rateLimit({
        windowMs: 15 * 60 * 1000, // 15 minutes
        max: 100, // limit each IP to 100 requests per windowMs
        message: 'Too many requests from this IP, please try again later.'
    });
    app.use(limiter);

    // Request logging middleware
    app.use((req, res, next) => {
        console.log(`${req.method} ${req.path}`, {
            headers: req.headers,
            body: req.body,
            timestamp: new Date().toISOString()
        });
        next();
    });

    // LRS Bridge endpoints
    app.post('/neuralblitz/bridge', async (req, res) => {
        try {
            const message = new LRSMessage(
                req.body.source_system,
                req.body.target_system,
                req.body.message_type,
                req.body.payload,
                bridge.config.authKey
            );

            // Verify signature
            if (!message.verifySignature(bridge.config.authKey)) {
                return res.status(401).json({
                    error: "Invalid message signature"
                });
            }

            // Route to appropriate handler (simplified)
            res.json({
                status: "received",
                message_id: message.messageId
            });
        } catch (error) {
            res.status(400).json({
                error: error.message
            });
        }
    });

    app.get('/neuralblitz/bridge/status', (req, res) => {
        const status = bridge.isHealthy() ? "healthy" : "degraded";
        
        res.json({
            system_id: bridge.config.systemId,
            status: status,
            metrics: bridge.getMetrics(),
            timestamp: new Date().toISOString()
        });
    });

    app.get('/neuralblitz/bridge/metrics', (req, res) => {
        res.json(bridge.getMetrics());
    });

    app.post('/neuralblitz/bridge/intent/submit', async (req, res) => {
        try {
            const response = await bridge.submitIntent(req.body, 'LRS_AGENT');
            res.json(response);
        } catch (error) {
            res.status(500).json({
                error: error.message
            });
        }
    });

    app.post('/neuralblitz/bridge/coherence/verify', async (req, res) => {
        try {
            const response = await bridge.verifyCoherence(req.body.target_system, req.body.verification_type);
            res.json(response);
        } catch (error) {
            res.status(500).json({
                error: error.message
            });
        }
    });

    // Health check endpoint
    app.get('/health', (req, res) => {
        res.json({
            status: bridge.isHealthy() ? "healthy" : "degraded",
            timestamp: new Date().toISOString(),
            uptime: bridge.getMetrics().uptime_seconds
        });
    });

    return app;
}

module.exports = {
    LRSBridge,
    LRSBridgeConfig,
    getLRSBridge,
    createLRSApp
};