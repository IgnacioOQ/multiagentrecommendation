# Inter-Chatbot Communication Protocol
- status: todo
- type: plan
- id: product.saas.features.inter_chatbot
- owner: product-manager
- priority: high
- estimate: 12w
- blocked_by: [product.saas.roadmap.phase2]
- context_dependencies: {"master_plan": "MASTER_PLAN.json", "conventions": "MD_CONVENTIONS.md"}
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
This plan defines the architecture and implementation roadmap for enabling **Inter-Chatbot Communication (ICC)** — the capability for our chatbot to interact with external chatbots from other companies. This creates a federated AI ecosystem where specialized agents can delegate tasks, share context, and collaborate across organizational boundaries.

**Strategic Value**:
1. **Extended Capabilities**: Access specialized services (legal bots, medical bots, translation bots) without building them in-house.
2. **B2B Integration**: Enable enterprise clients to connect their existing AI assistants with our platform.
3. **Network Effects**: Position the platform as a hub in an emerging inter-agent economy.

## Problem Statement
- status: active
- type: context
- id: product.saas.features.inter_chatbot.problem
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
Current AI chatbots operate as isolated silos. When a user asks a question outside the bot's domain, the typical response is "I can't help with that." Inter-chatbot communication solves this by allowing:

- **Delegation**: Our chatbot recognizes it lacks expertise and routes the query to a specialized external bot.
- **Collaboration**: Multiple bots work together on complex, multi-domain tasks.
- **Context Preservation**: User context flows seamlessly across bot boundaries without the user repeating themselves.

**Key Challenges**:
1. **Protocol Standardization**: No universal standard for bot-to-bot communication exists.
2. **Trust & Authentication**: How do bots verify each other's identity and capabilities?
3. **Context Translation**: Different bots use different internal representations.
4. **Privacy & Data Sovereignty**: What data can be shared across organizational boundaries?

## Architecture Overview
- status: active
- type: plan
- id: product.saas.features.inter_chatbot.architecture
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
The ICC system introduces three new architectural layers that integrate with the existing platform.

### ICC Gateway
- status: todo
- type: context
- id: product.saas.features.inter_chatbot.architecture.gateway
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
A dedicated service that handles all external bot communications. It sits between our Chatbot App and the outside world, acting as both a **client** (when we call external bots) and a **server** (when external bots call us).

**Responsibilities**:
- Protocol translation (adapt to different bot APIs)
- Authentication and authorization
- Rate limiting and abuse prevention
- Request/response logging for audit trails

### Bot Registry
- status: todo
- type: context
- id: product.saas.features.inter_chatbot.architecture.registry
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
A catalog of known external bots with their capabilities, endpoints, authentication requirements, and trust levels.

**Registry Entry Schema**:
```json
{
  "bot_id": "legal-advisor-acme",
  "provider": "ACME Legal Tech",
  "capabilities": ["contract_review", "compliance_check", "legal_qa"],
  "endpoint": "https://api.acme-legal.com/v1/chat",
  "auth_type": "oauth2",
  "trust_level": "verified",
  "rate_limits": {"rpm": 60, "daily": 1000},
  "data_policy": {"pii_allowed": false, "retention": "none"}
}
```

### Intent Router Enhancement
- status: todo
- type: context
- id: product.saas.features.inter_chatbot.architecture.router
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
Extends the existing Agent Orchestrator to detect when a query should be delegated externally. The router maintains a **capability map** that matches user intents to available bots (internal and external).

**Routing Decision Flow**:
1. Parse user intent
2. Check internal capability coverage
3. If gap detected → query Bot Registry for external matches
4. Apply trust/privacy filters
5. Route to best available handler (internal or external)

## Communication Protocol
- status: active
- type: protocol
- id: product.saas.features.inter_chatbot.protocol
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
Defines the message format and interaction patterns for bot-to-bot communication.

### Message Envelope
- status: todo
- type: protocol
- id: product.saas.features.inter_chatbot.protocol.envelope
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
All ICC messages use a standardized envelope that wraps the actual payload.

```json
{
  "icc_version": "1.0",
  "message_id": "uuid-v4",
  "timestamp": "ISO-8601",
  "sender": {
    "bot_id": "our-analyst-bot",
    "org_id": "our-company",
    "signature": "jwt-token"
  },
  "recipient": {
    "bot_id": "legal-advisor-acme",
    "org_id": "acme-legal"
  },
  "conversation": {
    "thread_id": "uuid-for-multi-turn",
    "parent_message_id": "previous-message-uuid",
    "context_hash": "sha256-of-shared-context"
  },
  "intent": {
    "action": "delegate_query",
    "capability_requested": "contract_review",
    "urgency": "normal"
  },
  "payload": {
    "type": "text|structured|file_reference",
    "content": "...",
    "attachments": []
  },
  "constraints": {
    "max_response_time_ms": 30000,
    "pii_handling": "redact",
    "response_format": "markdown"
  }
}
```

### Interaction Patterns
- status: todo
- type: protocol
- id: product.saas.features.inter_chatbot.protocol.patterns
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
Supported interaction models between bots:

1. **Request-Response (Synchronous)**: Simple query delegation. Our bot waits for a response.
2. **Request-Callback (Asynchronous)**: For long-running tasks. External bot calls back when done.
3. **Streaming**: Real-time token streaming for conversational handoffs.
4. **Multi-Turn Session**: Persistent conversation threads across multiple exchanges.

### Context Sharing Protocol
- status: todo
- type: protocol
- id: product.saas.features.inter_chatbot.protocol.context
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
Defines how conversation context is shared between bots without exposing raw user data.

**Context Levels**:
1. **Minimal**: Only the current query, no history.
2. **Summary**: LLM-generated summary of relevant conversation history.
3. **Selective**: Specific context items explicitly approved for sharing.
4. **Full**: Complete conversation thread (requires explicit user consent).

**Privacy-Preserving Techniques**:
- PII detection and redaction before transmission
- Differential privacy for aggregate statistics
- Tokenized references instead of raw identifiers

## Security Framework
- status: active
- type: guideline
- id: product.saas.features.inter_chatbot.security
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
Security is paramount when opening communication channels with external systems.

### Authentication & Authorization
- status: todo
- type: task
- id: product.saas.features.inter_chatbot.security.auth
- estimate: 2w
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
**Authentication Methods** (in order of preference):
1. **Mutual TLS (mTLS)**: Certificate-based identity for high-trust partners.
2. **OAuth 2.0 + JWT**: Standard token-based auth for API access.
3. **API Keys**: Simple shared secrets for low-sensitivity integrations.

**Authorization Model**:
- **Capability-Based Access**: Bots declare what they can do; we verify against registry.
- **Scope Limitations**: Each integration defines permitted data types and operations.
- **User Consent**: External delegation requires user opt-in (configurable per-bot).

### Trust Tiers
- status: todo
- type: context
- id: product.saas.features.inter_chatbot.security.trust
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
External bots are classified into trust tiers that determine interaction permissions:

| Tier | Name | Requirements | Permissions |
|------|------|--------------|-------------|
| 0 | Unknown | None | Blocked by default |
| 1 | Registered | API key + TOS acceptance | Read-only queries, no PII |
| 2 | Verified | Business verification + audit | Standard data sharing |
| 3 | Partner | Contract + security review + mTLS | Full integration, sensitive data |

### Threat Mitigation
- status: todo
- type: guideline
- id: product.saas.features.inter_chatbot.security.threats
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
**Identified Threats and Mitigations**:

1. **Prompt Injection via External Bot**: External bot returns malicious content.
   - *Mitigation*: Sanitize all external responses; treat as untrusted input.

2. **Data Exfiltration**: Malicious bot extracts sensitive user data.
   - *Mitigation*: PII redaction; data minimization; audit logging.

3. **Denial of Service**: External bot floods our system.
   - *Mitigation*: Per-bot rate limits; circuit breakers; timeout enforcement.

4. **Impersonation**: Attacker pretends to be a trusted bot.
   - *Mitigation*: mTLS certificates; signed messages; registry verification.

5. **Context Poisoning**: External bot injects false context into conversation.
   - *Mitigation*: Context isolation; explicit provenance tracking.

## Implementation Roadmap
- status: active
- type: plan
- id: product.saas.features.inter_chatbot.roadmap
- blocked_by: [product.saas.roadmap.phase2]
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
Phased implementation to deliver value incrementally while managing complexity.

### Phase A: Foundation (Outbound Only)
- status: todo
- type: plan
- id: product.saas.features.inter_chatbot.roadmap.phase_a
- estimate: 4w
- blocked_by: [product.saas.roadmap.phase2]
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
**Objective**: Enable our chatbot to call external bots (we are the client).

#### A.1: ICC Gateway (Client Mode)
- status: todo
- type: task
- id: product.saas.features.inter_chatbot.roadmap.phase_a.gateway
- estimate: 2w
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
Build the outbound communication service.

**Deliverables**:
- HTTP client with retry logic and circuit breakers
- Request/response logging and metrics
- Configurable timeout handling
- Support for REST and WebSocket protocols

**Tech Stack**: Python (httpx/aiohttp), deployed on Cloud Run.

#### A.2: Bot Registry (Read-Only)
- status: todo
- type: task
- id: product.saas.features.inter_chatbot.roadmap.phase_a.registry
- estimate: 1w
- blocked_by: [product.saas.features.inter_chatbot.roadmap.phase_a.gateway]
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
Implement the bot catalog as a configuration file (JSON/YAML) for initial rollout.

**Deliverables**:
- Registry schema definition
- CRUD operations for registry entries (admin only)
- Capability search/filtering API

#### A.3: Intent Router Integration
- status: todo
- type: task
- id: product.saas.features.inter_chatbot.roadmap.phase_a.router
- estimate: 1w
- blocked_by: [product.saas.features.inter_chatbot.roadmap.phase_a.registry]
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
Extend the existing Orchestrator to support external routing decisions.

**Deliverables**:
- Capability gap detection logic
- External routing decision module
- User notification when delegation occurs
- Response integration back into conversation

### Phase B: Inbound & Multi-Turn
- status: todo
- type: plan
- id: product.saas.features.inter_chatbot.roadmap.phase_b
- estimate: 4w
- blocked_by: [product.saas.features.inter_chatbot.roadmap.phase_a]
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
**Objective**: Allow external bots to call our chatbot; support multi-turn conversations.

#### B.1: ICC Gateway (Server Mode)
- status: todo
- type: task
- id: product.saas.features.inter_chatbot.roadmap.phase_b.server
- estimate: 2w
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
Expose our chatbot capabilities to external callers.

**Deliverables**:
- Public API endpoint for ICC messages
- Authentication middleware (OAuth2 + API key support)
- Rate limiting per caller
- Capability advertisement endpoint (machine-readable)

#### B.2: Conversation Threading
- status: todo
- type: task
- id: product.saas.features.inter_chatbot.roadmap.phase_b.threading
- estimate: 1w
- blocked_by: [product.saas.features.inter_chatbot.roadmap.phase_b.server]
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
Implement persistent conversation state across bot boundaries.

**Deliverables**:
- Thread ID generation and tracking
- Cross-bot session storage (Redis/Firestore)
- Context restoration on thread continuation
- Thread expiration and cleanup policies

#### B.3: Async Callback Support
- status: todo
- type: task
- id: product.saas.features.inter_chatbot.roadmap.phase_b.async
- estimate: 1w
- blocked_by: [product.saas.features.inter_chatbot.roadmap.phase_b.threading]
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
Enable asynchronous task delegation for long-running operations.

**Deliverables**:
- Webhook callback registration
- Task status polling endpoint
- Timeout and retry logic for callbacks
- User notification for async completions

### Phase C: Advanced Features
- status: todo
- type: plan
- id: product.saas.features.inter_chatbot.roadmap.phase_c
- estimate: 4w
- blocked_by: [product.saas.features.inter_chatbot.roadmap.phase_b]
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
**Objective**: Production hardening, trust management, and ecosystem features.

#### C.1: Trust Tier System
- status: todo
- type: task
- id: product.saas.features.inter_chatbot.roadmap.phase_c.trust
- estimate: 1w
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
Implement the tiered trust model for bot classification.

**Deliverables**:
- Trust tier assignment workflow
- Permission enforcement based on tier
- Audit logging for cross-tier access attempts
- Admin dashboard for trust management

#### C.2: Context Privacy Engine
- status: todo
- type: task
- id: product.saas.features.inter_chatbot.roadmap.phase_c.privacy
- estimate: 2w
- blocked_by: [product.saas.features.inter_chatbot.roadmap.phase_c.trust]
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
Build the privacy-preserving context sharing system.

**Deliverables**:
- PII detection model integration (spaCy/Presidio)
- Context summarization for external sharing
- User consent management UI
- Data lineage tracking

#### C.3: Bot Marketplace (Discovery)
- status: todo
- type: task
- id: product.saas.features.inter_chatbot.roadmap.phase_c.marketplace
- estimate: 1w
- blocked_by: [product.saas.features.inter_chatbot.roadmap.phase_c.privacy]
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
Create a discovery mechanism for users to find and enable external bot integrations.

**Deliverables**:
- Searchable bot catalog UI
- User-facing bot profiles with reviews/ratings
- One-click integration enablement
- Usage analytics per integration

## User Experience Design
- status: active
- type: plan
- id: product.saas.features.inter_chatbot.ux
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
How users interact with and perceive cross-bot functionality.

### Transparency Principles
- status: active
- type: guideline
- id: product.saas.features.inter_chatbot.ux.transparency
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
Users must always know when external bots are involved.

1. **Pre-Delegation Notice**: "I'll check with [External Bot] for this. Is that okay?"
2. **Attribution**: Responses from external bots are clearly labeled.
3. **Data Disclosure**: Users see what information was shared externally.
4. **Opt-Out**: Users can disable external integrations at any time.

### UI Components
- status: todo
- type: task
- id: product.saas.features.inter_chatbot.ux.components
- estimate: 1w
- blocked_by: [product.saas.features.inter_chatbot.roadmap.phase_a]
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
New UI elements to support ICC:

- **Integration Badge**: Visual indicator showing which bot provided a response.
- **Handoff Animation**: Smooth transition when switching between bots.
- **External Bot Card**: Preview of bot capabilities before delegation.
- **Privacy Summary**: Expandable view of shared data per interaction.

## Integration Examples
- status: active
- type: context
- id: product.saas.features.inter_chatbot.examples
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
Concrete scenarios demonstrating ICC value.

### Example 1: Legal Compliance Check
- status: active
- type: context
- id: product.saas.features.inter_chatbot.examples.legal
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
**Scenario**: User asks our Analyst bot about GDPR compliance for their data handling.

**Flow**:
1. User: "Is my customer data handling GDPR compliant?"
2. Our Bot: Detects legal domain → queries Bot Registry → finds "ACME Legal Bot"
3. Our Bot: "I can check with ACME Legal Advisor for a compliance review. They'll see a summary of your data practices. Proceed?"
4. User: "Yes"
5. Our Bot: Sends sanitized context to ACME Legal Bot
6. ACME Bot: Returns compliance assessment
7. Our Bot: "According to ACME Legal Advisor: [response with attribution]"

### Example 2: Multi-Bot Collaboration
- status: active
- type: context
- id: product.saas.features.inter_chatbot.examples.multibot
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
**Scenario**: Complex query requiring multiple specialized bots.

User: "Analyze my sales data, translate the report to Spanish, and check if our promotions comply with EU regulations."

**Flow**:
1. Our Analyst Bot: Performs sales analysis (internal capability)
2. Routes translation request → External Translation Bot
3. Routes compliance check → External Legal Bot
4. Aggregates results into unified response
5. User sees integrated output with clear attribution

## Metrics & Success Criteria
- status: active
- type: context
- id: product.saas.features.inter_chatbot.metrics
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
**Key Performance Indicators**:

| Metric | Target | Measurement |
|--------|--------|-------------|
| Delegation Success Rate | > 95% | Successful external calls / total attempts |
| Avg. External Response Time | < 5s | P50 latency for delegated queries |
| User Opt-In Rate | > 60% | Users enabling external integrations |
| Context Privacy Compliance | 100% | No PII leaked in external calls |
| Integration Uptime | > 99.5% | External bot availability |

**Qualitative Goals**:
- Users report seamless experience across bot boundaries
- Partner bots report easy integration process
- No security incidents related to ICC

## Open Questions & Decisions
- status: active
- type: log
- id: product.saas.features.inter_chatbot.decisions
- last_checked: 2026-01-25T12:00:00+01:00
<!-- content -->
Tracking key decisions and unresolved questions.

### Decision Log

| Date | Question | Decision | Rationale |
|------|----------|----------|-----------|
| 2026-01-25 | Standard protocol? | Custom JSON over HTTPS | No universal standard exists; design for future compatibility |
| 2026-01-25 | Default trust level? | Tier 0 (blocked) | Security-first; explicit opt-in required |

### Open Questions

1. **Billing Model**: How do we handle costs when external bots charge per query?
2. **SLA Propagation**: How do external bot SLAs affect our user-facing SLAs?
3. **Liability**: Who is responsible when an external bot provides incorrect information?
4. **Protocol Evolution**: How do we handle versioning and backward compatibility?
