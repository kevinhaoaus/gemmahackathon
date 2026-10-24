"""Capability handler modules. Every module here is imported at startup; each registers handlers
with @register_capability("<name>"). A node only runs a handler if it lists the capability in its
config `capabilities` (so one codebase can serve laptops, Pis and arm controllers)."""
