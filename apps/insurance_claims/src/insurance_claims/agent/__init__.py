"""The ReAct claims agent.

``loop.ClaimsAgent`` runs one tool-calling loop per caller turn. The model decides what to
ask and which tool to call; ``tools.ToolExecutor`` enforces every gate in code (which tools
exist in each phase, the three-field identity check, party-scoped data access, email consent),
and ``reply_guard.ReplyGuard`` (built on the checks in ``guardrails``) checks each reply before
the caller sees it.
"""
