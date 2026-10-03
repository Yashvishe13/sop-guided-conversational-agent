Design an Standard Operating Procedure (SOP) harness for an insurance claims support agent where the agent must follow a fixed business workflow, but still converse naturally.

At a high-level, the workflow has 4 phases: VERIFY_ID -> RESOLVE_INTENT -> PROCESS_CASE -> POST_PROCESS

The key challenge: different steps need different levels of freedom. Some steps require strict SOP control. For example, in VERIFY_ID, the agent must not disclose claim details or advance until identity is verified based on at least 3 PII info (Full name, DOB, Phone, Email, SSN last 4 digits). But it should still handle natural conversation, clarification questions, partial answers, refusals, and alternate identity fields.

Other steps allow freer LLM reasoning. For example, in RESOLVE_INTENT and PROCESS_CASE, the LLM can interpret messy user language, resolve ambiguity, answer grounded follow-up questions, and decide which bounded workflow path best matches the caller’s need. In POST_PROCESS, the agent should offer to send the customer an email summary of the conversation, including what was discussed, the claim status/outcome, and the major follow-up items or next steps. The customer must be able to choose whether to send the email or skip it.

Also, the system should be able to limit answers to only in-scope questions relevant to the insurance customer service scenario. Agent should reject answering any out of scope questions (e.g. what is RL?) politely. And ask to talk to human representatives if users keep retry on irrelevant questions.

The system should also remember useful information whenever the user says it, even if it belongs to a later phase. For example, if the caller says during verification, “I’m calling about my denied healthcare claim from January,” the agent must stay in VERIFY_ID, but store that intent and case hint for use after verification.

Sample test data is attached in the 'starter code' section. You can use it to test your harness.

Insurance Demo Test Case Caller says: I’m the policyholder. My name is Margaret Chen, policy POL-9921. I’m calling about my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.

Expected behavior: The agent extracts identity info and verifies the caller. It does not disclose claim details before verification. It remembers the denied-healthcare-January hint from the earlier utterance. After verification, it uses the remembered hint to resolve intent/case instead of asking from scratch. In PROCESS_CASE, it answers naturally but only from grounded claim/tool data. The LLM may interpret and phrase; the SOP still controls phase order, safety gates, and allowed actions. Success means the agent feels conversational without becoming a free-form chatbot: strict where the SOP demands it, flexible where reasoning is useful, and memory-aware across phase boundaries.

Bonus part: Emotional Support and SOP Recovery Design the agent to handle emotionally charged conversations like a real customer-service representative. The agent should be able to:

Recognize frustration, anxiety, anger, confusion, or refusal.
Respond with empathy and de-escalation before pushing the workflow forward.
Explain why required SOP steps matter, especially identity verification and consent.
Persuade the caller to continue without bypassing required gates.
Offer acceptable alternatives when possible, such as different ID fields or human transfer. Know when to stop persuading and escalate to a human. Example: Caller: I already told you who I am. This is ridiculous. Just tell me why my claim was denied. Expected behavior: Acknowledge the frustration. Explain that claim details are protected and require verification first. Offer the allowed verification options. Keep the conversation moving toward SOP completion. Do not disclose claim details or skip verification.
