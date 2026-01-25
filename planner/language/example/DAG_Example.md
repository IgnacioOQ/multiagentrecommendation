# Software Release Cycle
- id: release_cycle
- context_dependencies: { "conventions": "../../MD_CONVENTIONS.md" }- status: active
- type: context
<!-- content -->

## Development
- id: phase.dev
- status: in-progress
<!-- content -->

### Backend Implementation
- id: dev.backend
- status: done
<!-- content -->

### Frontend Implementation
- id: dev.frontend
- status: in-progress
- blocked_by: [dev.backend]
<!-- content -->

## Testing
- id: phase.testing
- status: todo
- blocked_by: [phase.dev]
<!-- content -->

### Unit Tests
- id: test.unit
- status: todo
- blocked_by: [dev.backend, dev.frontend]
<!-- content -->

### Integration Tests
- id: test.integration
- status: todo
- blocked_by: [test.unit]
<!-- content -->

## Deployment
- id: phase.deploy
- status: blocked
- blocked_by: [phase.testing]
<!-- content -->
