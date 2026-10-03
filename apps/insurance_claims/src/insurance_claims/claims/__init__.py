"""Trusted claim data and identity checks.

Everything here is deterministic and model-free: fixtures are validated at startup,
repositories only ever return the verified party's own claims, ``verification`` decides
whether three identity fields match one policyholder, and ``evidence`` builds the fact set
that replies are grounded against.
"""
