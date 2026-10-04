"""The ReAct claims agent.

``sop`` loads the SOP from ``sop.toml``: the phase order, which tools exist in each phase, and the
only allowed phase changes. ``loop.ClaimsAgent`` runs one tool-calling loop per caller turn. The model decides what to
ask and which tool to call; ``tools.ToolExecutor`` enforces every gate in code (which tools
exist in each phase, the three-field identity check, party-scoped data access, email consent),
and ``reply_guard.ReplyGuard`` (built on the checks in ``guardrails``) checks each reply before
the caller sees it. ``guard.Guard`` is an independent model reviewer that code consults where a
rule depends on understanding language: who is speaking, what the caller consented to, whether
an email summary is true, and whether a reply stays in scope.
"""
