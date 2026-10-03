/*
 * Claims support chat client.
 *
 * Security rules this file follows:
 *  - Server text is rendered with textContent only (never innerHTML).
 *  - No inline styles or handlers (CSP default-src 'self').
 *  - The CSRF token lives only in memory; the session cookie is HttpOnly.
 *  - Only the opaque session id is stored in localStorage.
 */
(function () {
  "use strict";

  var STORAGE_KEY = "ic_session_id";
  var MAX_CHARS = 2000;
  var COUNTER_FROM = 1700;
  var NEAR_LIMIT = 100;
  var REQUEST_TIMEOUT_MS = 180000;
  var SESSION_ID_RE = /^[A-Za-z0-9_-]{8,128}$/;
  var PHASES = ["VERIFY_ID", "RESOLVE_INTENT", "PROCESS_CASE", "POST_PROCESS"];
  var STEP_STATUS_TEXT = { done: "completed", active: "current step", pending: "not started" };
  // Must match the user-message labels the server records for button actions.
  var ACTION_LABELS = { email_send: "Send the email summary", email_skip: "Skip the email" };
  var MESSAGE_KINDS = { chat: true, email_offer: true, notice: true };
  var RETRYABLE_STATUS = { 0: true, 408: true, 429: true, 500: true, 502: true, 503: true, 504: true };
  // The only 409s that mean "try the same turn again": the server is busy with, or raced, another turn.
  var RETRYABLE_CONFLICT = { turn_in_progress: true, version_conflict: true };
  // Failures after which the server may still have processed the turn (lost response, timeout, crash).
  var UNKNOWN_OUTCOME_STATUS = { 0: true, 408: true, 500: true, 502: true, 503: true, 504: true };
  var SESSION_GONE_STATUS = { 401: true, 404: true, 410: true };
  // The stored conversation cannot be continued whatever the status: start a new one, never retry.
  var SESSION_GONE_CODE = { session_not_found: true, session_expired: true, session_unrecoverable: true };

  var COPY = {
    queued: "Your message will be sent as soon as the agent is ready.",
    connecting: "Connecting",
    typing: "Agent is typing",
    stillWorking: "Still working on your last message. Try again in a moment.",
    network: "We could not reach the server. Check your connection and try again.",
    badResponse: "The server sent a response we could not read. Try again.",
    sessionGone: "This conversation has ended or expired. Start a new conversation to continue.",
    unrecoverable: "This conversation could not be restored. Start a new conversation to continue.",
    startFailed: "We could not start a conversation.",
    tooLong: "Messages can be up to " + MAX_CHARS + " characters. Shorten your message and send it again.",
    closed: "This conversation has ended. Start a new conversation to continue.",
    placeholder: "Type your message",
    confirmNew: "Start a new conversation? The current chat will be cleared from this browser.",
    handoffRequested: "You asked for a human representative. Your request is noted on this conversation.",
    handoffOffered: "A human representative can help with this. Ask for one at any time.",
    locked: "Identity checks are paused for this conversation. A human representative can help you from here.",
    notSent: "Not sent",
    unconfirmed: "May not have been sent",
    sending: "Sending",
    assistantSaid: "Assistant said",
    youSaid: "You said",
    tryAgain: "Try again",
    startNew: "Start new conversation"
  };

  // ------------------------------------------------------------------ elements

  function byId(id) {
    var node = document.getElementById(id);
    if (!node) {
      throw new Error("Missing element #" + id);
    }
    return node;
  }

  var el = {
    newConversation: byId("new-conversation"),
    statusRow: byId("status-row"),
    verifiedBadge: byId("verified-badge"),
    caseChip: byId("case-chip"),
    progress: byId("progress"),
    handoffNotice: byId("handoff-notice"),
    handoffText: byId("handoff-text"),
    scroller: byId("transcript-scroll"),
    transcript: byId("transcript"),
    typing: byId("typing"),
    typingText: byId("typing-text"),
    liveStatus: byId("live-status"),
    errorBanner: byId("error-banner"),
    errorMessage: byId("error-message"),
    errorRetry: byId("error-retry"),
    errorDismiss: byId("error-dismiss"),
    composer: byId("composer"),
    input: byId("composer-input"),
    send: byId("send-button"),
    charCount: byId("char-count")
  };

  // ------------------------------------------------------------------ app state

  var app = {
    sessionId: null,
    csrfToken: null, // memory only
    state: null, // last state.public_view() from the server
    ready: false, // a session is loaded
    busy: false, // a request is in flight
    epoch: 0, // bumps on reset; stale responses are ignored
    pending: null, // failed turn awaiting retry: { clientTurnId, body, bubble, userText, epoch, csrfRetried, uncertain }
    inflight: null, // AbortController of the tracked request
    keys: new Set(), // turn_index|role|text of rendered server messages
    resyncNeeded: false, // the transcript may differ from server history; rebuild it after the next turn
    historySeq: 0, // bumps when a turn is sent or answered; a GET that overlapped one is stale
    retryAction: null,
    messageSeq: 0
  };

  // ------------------------------------------------------------------ storage (best effort)

  function readStoredId() {
    try {
      var value = window.localStorage.getItem(STORAGE_KEY);
      return value && SESSION_ID_RE.test(value) ? value : null;
    } catch (err) {
      return null;
    }
  }

  function storeId(id) {
    try {
      window.localStorage.setItem(STORAGE_KEY, id);
    } catch (err) {
      /* storage blocked: the chat still works, it just will not resume after a refresh */
    }
  }

  function clearStoredId() {
    try {
      window.localStorage.removeItem(STORAGE_KEY);
    } catch (err) {
      /* ignore */
    }
  }

  // ------------------------------------------------------------------ ids

  function randomHex(bytes) {
    var buf = new Uint8Array(bytes);
    if (window.crypto && typeof window.crypto.getRandomValues === "function") {
      window.crypto.getRandomValues(buf);
    } else {
      for (var i = 0; i < bytes; i += 1) {
        buf[i] = Math.floor(Math.random() * 256);
      }
    }
    var out = "";
    for (var j = 0; j < buf.length; j += 1) {
      out += (buf[j] + 0x100).toString(16).slice(1);
    }
    return out;
  }

  /** A fresh client_turn_id: crypto.randomUUID, else a v4-shaped id built from random hex. */
  function newTurnId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      try {
        return window.crypto.randomUUID();
      } catch (err) {
        /* insecure context: fall through */
      }
    }
    var h = randomHex(16);
    var variant = ((parseInt(h.charAt(16), 16) & 0x3) | 0x8).toString(16);
    return h.slice(0, 8) + "-" + h.slice(8, 12) + "-4" + h.slice(13, 16) + "-" + variant + h.slice(17, 20) + "-" + h.slice(20, 32);
  }

  // ------------------------------------------------------------------ HTTP

  function ApiError(status, code, message, requestId) {
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.message = message;
    this.requestId = requestId || null;
  }
  ApiError.prototype = Object.create(Error.prototype);
  ApiError.prototype.constructor = ApiError;

  function networkError() {
    return new ApiError(0, "network_error", COPY.network);
  }

  /** The stored conversation cannot continue (unknown, expired, foreign, or unreadable checkpoint). */
  function isSessionGone(err) {
    return err instanceof ApiError &&
      (SESSION_GONE_STATUS[err.status] === true || SESSION_GONE_CODE[err.code] === true);
  }

  function isStillWorking(err) {
    return err instanceof ApiError && err.status === 409 && RETRYABLE_CONFLICT[err.code] === true;
  }

  /** Safe to resend the same turn (same client_turn_id, so the server dedupes). */
  function isRetryable(err) {
    return err instanceof ApiError && (RETRYABLE_STATUS[err.status] === true || isStillWorking(err));
  }

  function sessionPath(id) {
    return "/api/sessions/" + encodeURIComponent(id);
  }

  function readErrorEnvelope(status, data) {
    var envelope = data && typeof data === "object" && data.error && typeof data.error === "object" ? data.error : null;
    var code = envelope && typeof envelope.code === "string" ? envelope.code : "http_" + status;
    var message = envelope && typeof envelope.message === "string" && envelope.message.trim()
      ? envelope.message.trim()
      : "Something went wrong (status " + status + "). Try again.";
    var requestId = envelope && typeof envelope.request_id === "string" ? envelope.request_id : null;
    return new ApiError(status, code, message, requestId);
  }

  /**
   * JSON request against the same-origin API. Resolves with the parsed body (null for 204).
   * Rejects with ApiError; status 0 means the network failed, timed out, or was aborted.
   * options.csrf: false omits X-CSRF-Token; options.csrfToken overrides the token;
   * options.track: true lets a reset abort this request.
   */
  function request(method, path, body, options) {
    var opts = options || {};
    var headers = { Accept: "application/json" };
    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
    }
    var token = opts.csrfToken || app.csrfToken;
    if (opts.csrf !== false && (method === "POST" || method === "DELETE") && token) {
      headers["X-CSRF-Token"] = token;
    }
    var controller = typeof AbortController === "function" ? new AbortController() : null;
    if (opts.track && controller) {
      app.inflight = controller;
    }
    var timer = controller ? window.setTimeout(function () { controller.abort(); }, REQUEST_TIMEOUT_MS) : null;
    var init = { method: method, headers: headers, credentials: "same-origin", cache: "no-store" };
    if (body !== undefined) {
      init.body = JSON.stringify(body);
    }
    if (controller) {
      init.signal = controller.signal;
    }

    function settle() {
      if (timer !== null) {
        window.clearTimeout(timer);
      }
      if (controller && app.inflight === controller) {
        app.inflight = null;
      }
    }

    return window.fetch(path, init).then(
      function (res) {
        if (res.status === 204) {
          settle();
          return null;
        }
        return res.text().then(
          function (raw) {
            settle();
            var data = null;
            if (raw) {
              try {
                data = JSON.parse(raw);
              } catch (err) {
                data = null;
              }
            }
            if (!res.ok) {
              throw readErrorEnvelope(res.status, data);
            }
            if (!data || typeof data !== "object" || Array.isArray(data)) {
              throw new ApiError(res.status, "bad_response", COPY.badResponse);
            }
            return data;
          },
          function () {
            settle();
            throw networkError();
          }
        );
      },
      function () {
        settle();
        throw networkError();
      }
    );
  }

  // ------------------------------------------------------------------ message rendering

  function isValidMessage(m) {
    return Boolean(
      m && typeof m === "object" &&
      (m.role === "user" || m.role === "assistant") &&
      typeof m.text === "string" &&
      typeof m.turn_index === "number" && isFinite(m.turn_index)
    );
  }

  function messageKey(turnIndex, role, text) {
    return turnIndex + "|" + role + "|" + text;
  }

  function messageKind(m) {
    return typeof m.kind === "string" && MESSAGE_KINDS[m.kind] === true ? m.kind : "chat";
  }

  function buildMessage(role, kind, text) {
    app.messageSeq += 1;
    var item = document.createElement("li");
    item.className = "msg msg--" + role + " msg--" + kind;

    var bubble = document.createElement("div");
    bubble.className = "msg__bubble";

    var who = document.createElement("span");
    who.className = "visually-hidden";
    who.textContent = (role === "user" ? COPY.youSaid : COPY.assistantSaid) + " ";

    var para = document.createElement("p");
    para.className = "msg__text";
    para.id = "msg-text-" + app.messageSeq;
    para.textContent = text;

    bubble.appendChild(who);
    bubble.appendChild(para);
    item.appendChild(bubble);
    return item;
  }

  var pinnedToEnd = true; // the reader is at the bottom of the transcript
  var selfScrolled = false; // the next scroll event came from pin(), not the reader

  function pin() {
    var before = el.scroller.scrollTop;
    el.scroller.scrollTop = el.scroller.scrollHeight;
    if (el.scroller.scrollTop !== before) {
      selfScrolled = true;
    }
  }

  function scrollToEnd() {
    pin();
    pinnedToEnd = true;
  }

  /** Keep the latest message in view when the scroller shrinks (banner, notice, keyboard, textarea growth). */
  function watchScroller() {
    el.scroller.addEventListener("scroll", function () {
      if (selfScrolled) {
        selfScrolled = false;
        return;
      }
      var gap = el.scroller.scrollHeight - el.scroller.scrollTop - el.scroller.clientHeight;
      pinnedToEnd = gap < 48;
    }, { passive: true });
    if (typeof ResizeObserver !== "function") {
      return;
    }
    var observer = new ResizeObserver(function () {
      if (pinnedToEnd) {
        pin();
      }
    });
    observer.observe(el.scroller);
    observer.observe(el.transcript);
  }

  /** Append server messages in order, skipping any already shown (turn_index + role + text). */
  function appendMessages(messages) {
    if (!Array.isArray(messages)) {
      return;
    }
    messages.forEach(function (m) {
      if (!isValidMessage(m)) {
        return;
      }
      var key = messageKey(m.turn_index, m.role, m.text);
      if (app.keys.has(key)) {
        return;
      }
      app.keys.add(key);
      el.transcript.appendChild(buildMessage(m.role, messageKind(m), m.text));
    });
    scrollToEnd();
  }

  /** Rebuild the transcript from server history; an unsent local turn stays at the end. */
  function renderHistory(messages) {
    el.transcript.setAttribute("aria-busy", "true");
    el.transcript.textContent = "";
    app.keys = new Set();
    appendMessages(messages);
    if (app.pending && app.pending.bubble) {
      el.transcript.appendChild(app.pending.bubble);
    }
    el.transcript.removeAttribute("aria-busy");
    scrollToEnd();
  }

  var BUBBLE_STATES = {
    sending: { cls: "is-sending", text: COPY.sending },
    failed: { cls: "is-failed", text: COPY.notSent }, // the server did not take the turn
    unconfirmed: { cls: "is-unconfirmed", text: COPY.unconfirmed } // the server may have taken it
  };

  /** stateName: "sent", "sending", "failed", or "unconfirmed". */
  function setBubbleState(bubble, stateName) {
    if (!bubble) {
      return;
    }
    bubble.classList.remove("is-sending", "is-failed", "is-unconfirmed");
    var meta = bubble.querySelector(".msg__meta");
    var look = BUBBLE_STATES[stateName];
    if (!look) {
      if (meta) {
        meta.remove();
      }
      return;
    }
    bubble.classList.add(look.cls);
    if (!meta) {
      meta = document.createElement("p");
      meta.className = "msg__meta";
      bubble.appendChild(meta);
    }
    meta.textContent = look.text;
  }

  function addOptimisticBubble(text) {
    var item = buildMessage("user", "chat", text);
    setBubbleState(item, "sending");
    el.transcript.appendChild(item);
    scrollToEnd();
    return item;
  }

  // ------------------------------------------------------------------ state rendering

  function deriveSteps(phase) {
    var current = PHASES.indexOf(phase);
    return PHASES.map(function (p, i) {
      var status = current < 0 || i > current ? "pending" : i < current ? "done" : "active";
      return { phase: p, status: status };
    });
  }

  function renderProgress(state) {
    var steps = Array.isArray(state.steps) ? state.steps : deriveSteps(state.phase);
    var byPhase = {};
    steps.forEach(function (s) {
      if (s && typeof s.phase === "string" && Object.prototype.hasOwnProperty.call(STEP_STATUS_TEXT, s.status)) {
        byPhase[s.phase] = s.status;
      }
    });
    Array.prototype.forEach.call(el.progress.querySelectorAll(".progress__step"), function (item) {
      var status = byPhase[item.dataset.phase] || "pending";
      item.classList.remove("is-done", "is-active", "is-pending");
      item.classList.add("is-" + status);
      if (status === "active") {
        item.setAttribute("aria-current", "step");
      } else {
        item.removeAttribute("aria-current");
      }
      var label = item.querySelector(".progress__status");
      if (label) {
        label.textContent = STEP_STATUS_TEXT[status];
      }
    });
  }

  function renderBadges(state) {
    var verified = state.verified === true;
    el.verifiedBadge.hidden = !verified;
    var caseInfo = state.case && typeof state.case === "object" ? state.case : null;
    var caseId = verified && caseInfo && typeof caseInfo.case_id === "string" ? caseInfo.case_id : "";
    el.caseChip.textContent = caseId ? "Claim " + caseId : "";
    el.caseChip.hidden = !caseId;
    el.statusRow.hidden = !verified && !caseId;
  }

  function renderNotice(state) {
    var handoff = state.handoff && typeof state.handoff === "object" ? state.handoff : {};
    var text = "";
    if (handoff.requested === true) {
      text = COPY.handoffRequested;
    } else if (state.verification_locked === true) {
      text = COPY.locked;
    } else if (handoff.offered === true) {
      text = COPY.handoffOffered;
    }
    if (el.handoffText.textContent !== text) {
      el.handoffText.textContent = text;
    }
    el.handoffNotice.hidden = !text;
  }

  function isClosed() {
    return Boolean(app.state && app.state.closed === true);
  }

  function emailOfferActive() {
    var email = app.state && app.state.email;
    return Boolean(email && typeof email === "object" && email.offer_active === true && !isClosed());
  }

  function emailButtonsDisabled() {
    return app.busy || !app.ready || app.pending !== null;
  }

  /** Place Send email / Skip under the latest email offer while the offer is active. */
  function renderEmailActions() {
    Array.prototype.forEach.call(el.transcript.querySelectorAll(".email-actions"), function (node) {
      node.remove();
    });
    if (!emailOfferActive()) {
      return;
    }
    var offers = el.transcript.querySelectorAll(".msg--assistant.msg--email_offer");
    var anchor = offers.length ? offers[offers.length - 1] : null;
    if (!anchor) {
      var assistants = el.transcript.querySelectorAll(".msg--assistant");
      anchor = assistants.length ? assistants[assistants.length - 1] : null;
    }
    if (!anchor) {
      return;
    }
    var describedBy = anchor.querySelector(".msg__text");
    var group = document.createElement("div");
    group.className = "email-actions";
    group.setAttribute("role", "group");
    group.setAttribute("aria-label", "Email summary choice");
    group.appendChild(emailButton("email_send", "Send email", "btn btn--primary", describedBy));
    group.appendChild(emailButton("email_skip", "Skip", "btn btn--quiet", describedBy));
    anchor.appendChild(group);
    scrollToEnd();
  }

  function emailButton(action, label, className, describedBy) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = className;
    button.textContent = label;
    button.dataset.action = action;
    button.disabled = emailButtonsDisabled();
    if (describedBy && describedBy.id) {
      button.setAttribute("aria-describedby", describedBy.id);
    }
    button.addEventListener("click", onEmailAction);
    return button;
  }

  function applyState(state) {
    if (!state || typeof state !== "object") {
      return;
    }
    app.state = state;
    renderProgress(state);
    renderBadges(state);
    renderNotice(state);
    renderEmailActions();
    updateControls();
  }

  var INITIAL_STATE = { phase: "VERIFY_ID", steps: deriveSteps("VERIFY_ID"), verified: false, handoff: {}, email: {}, closed: false };

  // ------------------------------------------------------------------ controls

  function updateControls() {
    var closed = isClosed();
    el.input.disabled = closed || (!app.ready && !app.busy); // typeable while reconnecting, not in a dead session
    el.input.placeholder = closed ? COPY.closed : COPY.placeholder;
    el.send.disabled = !app.ready || app.busy || closed || el.input.value.trim().length === 0;
    var disableEmail = emailButtonsDisabled();
    Array.prototype.forEach.call(el.transcript.querySelectorAll(".email-actions button"), function (b) {
      b.disabled = disableEmail;
    });
    updateCounter();
  }

  function updateCounter() {
    var length = el.input.value.length;
    if (length < COUNTER_FROM) {
      el.charCount.hidden = true;
      el.charCount.textContent = "";
      el.charCount.classList.remove("is-near-limit");
      return;
    }
    var left = Math.max(0, MAX_CHARS - length);
    el.charCount.hidden = false;
    el.charCount.textContent = left === 1 ? "1 character left" : left + " characters left";
    el.charCount.classList.toggle("is-near-limit", left <= NEAR_LIMIT);
  }

  function flushQueuedSend() {
    if (app.queuedSend && app.ready && !app.busy && !isClosed()) {
      app.queuedSend = false;
      if (el.input.value.trim()) {
        sendText();
      }
    }
  }

  function setBusy(busy, label) {
    app.busy = busy;
    if (!busy) {
      window.setTimeout(flushQueuedSend, 0);
    }
    el.typing.hidden = !busy;
    el.typingText.textContent = label || COPY.typing;
    el.liveStatus.textContent = busy ? label || COPY.typing : "";
    if (busy) {
      scrollToEnd();
    }
    updateControls();
  }

  function focusInput() {
    if (!el.input.disabled) {
      el.input.focus();
    }
  }

  function focusWasLost() {
    var active = document.activeElement;
    return !active || active === document.body || !active.isConnected;
  }

  // ------------------------------------------------------------------ errors

  function errorText(err) {
    if (!(err instanceof ApiError)) {
      return COPY.network;
    }
    var text = isStillWorking(err) ? COPY.stillWorking : err.message;
    return err.requestId ? text + " Reference " + err.requestId + "." : text;
  }

  function showError(message, retry, retryLabel) {
    var lostFocus = focusWasLost();
    el.errorMessage.textContent = message;
    app.retryAction = typeof retry === "function" ? retry : null;
    el.errorRetry.hidden = app.retryAction === null;
    el.errorRetry.textContent = retryLabel || COPY.tryAgain;
    el.errorBanner.hidden = false;
    if (lostFocus) {
      (app.retryAction ? el.errorRetry : el.errorDismiss).focus();
    }
  }

  function hideError() {
    el.errorBanner.hidden = true;
    el.errorMessage.textContent = "";
    app.retryAction = null;
  }

  // ------------------------------------------------------------------ session lifecycle

  function acceptSession(data) {
    if (!data || typeof data.session_id !== "string" || !SESSION_ID_RE.test(data.session_id)) {
      throw new ApiError(200, "bad_response", COPY.badResponse);
    }
    app.sessionId = data.session_id;
    if (typeof data.csrf_token === "string" && data.csrf_token) {
      app.csrfToken = data.csrf_token;
    }
    storeId(data.session_id);
    app.ready = true;
    renderHistory(data.messages);
    applyState(data.state);
    scrollToEnd();
  }

  function createSession(options) {
    var opts = options || {};
    var epoch = app.epoch;
    hideError();
    setBusy(true, COPY.connecting);
    return request("POST", "/api/sessions", {}, { csrf: false, track: true })
      .then(function (data) {
        if (epoch === app.epoch) {
          acceptSession(data);
        }
      })
      .catch(function (err) {
        if (epoch !== app.epoch) {
          return;
        }
        app.ready = false;
        showError(COPY.startFailed + " " + errorText(err), function () {
          createSession({ focus: true });
        });
      })
      .then(function () {
        if (epoch !== app.epoch) {
          return;
        }
        setBusy(false);
        if (opts.focus) {
          focusInput();
        }
      });
  }

  /** On load: resume the stored session, or start a new one when it is missing or gone. */
  function boot() {
    var stored = readStoredId();
    if (!stored) {
      return createSession();
    }
    var epoch = app.epoch;
    hideError();
    setBusy(true, COPY.connecting);
    return request("GET", sessionPath(stored), undefined, { track: true })
      .then(function (data) {
        if (epoch !== app.epoch) {
          return undefined;
        }
        if (data.session_id !== stored) {
          throw new ApiError(200, "bad_response", COPY.badResponse);
        }
        acceptSession(data);
        setBusy(false);
        return undefined;
      })
      .catch(function (err) {
        if (epoch !== app.epoch) {
          return undefined;
        }
        setBusy(false);
        // Unknown, expired, foreign, unreadable, or malformed stored id: start fresh instead of retrying forever.
        var gone = isSessionGone(err) || (err instanceof ApiError &&
          (err.status === 400 || err.status === 403 || err.status === 422 || err.code === "bad_response"));
        if (gone) {
          clearStoredId();
          return createSession();
        }
        showError(errorText(err), boot);
        return undefined;
      });
  }

  /**
   * Re-read the session for a fresh CSRF token, state, and history, and rebuild the transcript from it
   * (server history is the truth: hidden or unprocessed messages drop out). Rejects with ApiError.
   */
  function refreshSession() {
    var epoch = app.epoch;
    var seq = app.historySeq;
    return request("GET", sessionPath(app.sessionId)).then(function (data) {
      if (epoch !== app.epoch) {
        return;
      }
      if (seq !== app.historySeq) {
        // A turn was sent or answered meanwhile, so this history may be older than the transcript.
        if (data && data.session_id === app.sessionId && typeof data.csrf_token === "string" && data.csrf_token) {
          app.csrfToken = data.csrf_token;
        }
        app.resyncNeeded = true;
        return;
      }
      acceptSession(data);
      app.resyncNeeded = false;
    });
  }

  function resetLocalState(keepInput) {
    app.epoch += 1;
    if (app.inflight) {
      app.inflight.abort();
      app.inflight = null;
    }
    app.sessionId = null;
    app.csrfToken = null;
    app.state = null;
    app.ready = false;
    app.pending = null;
    app.resyncNeeded = false;
    app.keys = new Set();
    el.transcript.textContent = "";
    if (!keepInput) {
      el.input.value = "";
    }
    hideError();
    setBusy(false);
    applyState(INITIAL_STATE);
  }

  /**
   * Start over: confirm, best-effort DELETE of the old session, then POST /api/sessions.
   * options.skipConfirm, options.skipDelete, options.keepInput.
   */
  function startNewConversation(options) {
    var opts = options || {};
    var hasConversation = app.ready && app.sessionId !== null;
    if (hasConversation && !opts.skipConfirm && !window.confirm(COPY.confirmNew)) {
      return;
    }
    var oldId = app.sessionId;
    var oldCsrf = app.csrfToken;
    resetLocalState(Boolean(opts.keepInput));
    clearStoredId();
    if (!oldId || !oldCsrf || opts.skipDelete) {
      createSession({ focus: true });
      return;
    }
    var epoch = app.epoch;
    setBusy(true, COPY.connecting);
    request("DELETE", sessionPath(oldId), undefined, { csrfToken: oldCsrf, track: true })
      .catch(function () {
        return null; // the old session expires on its own; never block a fresh start
      })
      .then(function () {
        if (epoch === app.epoch) {
          createSession({ focus: true });
        }
      });
  }

  // ------------------------------------------------------------------ turns

  function newTurn(bubbleText, userText, body) {
    var turn = {
      clientTurnId: newTurnId(),
      body: null,
      bubble: addOptimisticBubble(bubbleText),
      userText: userText,
      epoch: app.epoch,
      csrfRetried: false,
      uncertain: false // an attempt failed in a way the server may still have processed it
    };
    body.client_turn_id = turn.clientTurnId;
    turn.body = body;
    return turn;
  }

  /**
   * A new send replaces a failed turn. The failed turn is never resent (it may be an action the
   * user moved away from), and the server may already have processed it, so the transcript is
   * rebuilt from server history after the next turn: a processed turn reappears with its reply,
   * an unprocessed one drops out.
   */
  function supersedeFailedTurn() {
    if (app.pending && !app.busy) {
      app.pending = null;
      app.resyncNeeded = true;
      hideError();
    }
  }

  function sendText() {
    if (isClosed()) {
      return;
    }
    if (!app.ready || app.busy) {
      // Not dropped: remember it and send as soon as the session is ready or the reply arrives.
      // The server dedupes by client_turn_id, so a queued send is processed at most once.
      if (el.input.value.trim()) {
        app.queuedSend = true;
        el.liveStatus.textContent = COPY.queued;
      }
      return;
    }
    app.queuedSend = false;
    var text = el.input.value.trim();
    if (!text) {
      return;
    }
    if (text.length > MAX_CHARS) {
      showError(COPY.tooLong);
      return;
    }
    supersedeFailedTurn();
    el.input.value = "";
    submitTurn(newTurn(text, text, { text: text }));
  }

  function onEmailAction(event) {
    var action = event.currentTarget.dataset.action;
    if (!Object.prototype.hasOwnProperty.call(ACTION_LABELS, action) || emailButtonsDisabled() || !emailOfferActive()) {
      return;
    }
    Array.prototype.forEach.call(el.transcript.querySelectorAll(".email-actions button"), function (b) {
      b.disabled = true;
    });
    submitTurn(newTurn(ACTION_LABELS[action], null, { action: action }));
  }

  function submitTurn(turn) {
    app.pending = turn;
    hideError();
    setBubbleState(turn.bubble, "sending");
    if (!turn.bubble.isConnected) {
      el.transcript.appendChild(turn.bubble);
    }
    setBusy(true, COPY.typing);
    app.historySeq += 1;
    request("POST", sessionPath(app.sessionId) + "/messages", turn.body, { track: true })
      .then(function (data) {
        if (turn.epoch !== app.epoch) {
          return;
        }
        app.pending = null;
        acceptTurn(turn, data);
        if (!app.resyncNeeded) {
          setBusy(false);
          focusInput();
          return;
        }
        // Stay busy so no new turn can race the resync. A failed resync is retried after the next turn.
        refreshSession()
          .catch(function () { return null; })
          .then(function () {
            if (turn.epoch !== app.epoch) {
              return;
            }
            setBusy(false);
            focusInput();
          });
      })
      .catch(function (err) {
        if (turn.epoch !== app.epoch) {
          return;
        }
        setBusy(false);
        handleTurnError(turn, err);
      });
  }

  function acceptTurn(turn, data) {
    var wasVerified = Boolean(app.state && app.state.verified === true);
    app.historySeq += 1;
    setBubbleState(turn.bubble, "sent");
    if (typeof data.turn_index === "number") {
      var shown = turn.bubble.querySelector(".msg__text");
      var key = messageKey(data.turn_index, "user", shown ? shown.textContent : "");
      if (app.keys.has(key)) {
        turn.bubble.remove(); // a resync already rendered this turn from history
      } else {
        app.keys.add(key);
      }
    }
    appendMessages(data.messages);
    applyState(data.state);
    if (wasVerified && !(app.state && app.state.verified === true)) {
      // Identity was reset (idle expiry, caller change): the server may now hide the earlier,
      // verified part of the transcript, so rebuild it from server history instead of keeping it.
      app.resyncNeeded = true;
    }
    scrollToEnd(); // notices and badges may have changed the layout
  }

  function retryPending() {
    var turn = app.pending;
    if (!turn || app.busy || turn.epoch !== app.epoch) {
      return;
    }
    submitTurn(turn);
  }

  /** Remove a turn that will not be retried; give typed text back for editing. */
  function dropTurn(turn, restoreText) {
    if (app.pending === turn) {
      app.pending = null;
    }
    turn.bubble.remove();
    if (restoreText && turn.userText && !el.input.value) {
      el.input.value = turn.userText;
    }
    renderEmailActions();
    updateControls();
  }

  /** Dismissing a failed turn cancels it, then quietly resyncs in case the server did process it. */
  function dismissPending() {
    var turn = app.pending;
    if (!turn || app.busy) {
      return;
    }
    dropTurn(turn, true);
    resyncNow();
  }

  /** Rebuild the transcript from server history now; if that fails or races a turn, after the next turn. */
  function resyncNow() {
    app.resyncNeeded = true;
    if (app.ready && app.sessionId && !app.busy) {
      refreshSession().catch(function () { return null; });
    }
  }

  function handleTurnError(turn, err) {
    var apiErr = err instanceof ApiError ? err : networkError();

    if (isSessionGone(apiErr)) {
      // Never retry into a conversation that cannot continue: offer a fresh one, keeping typed text.
      dropTurn(turn, true);
      app.ready = false;
      updateControls();
      var goneText = apiErr.code === "session_unrecoverable" ? COPY.unrecoverable : COPY.sessionGone;
      showError(apiErr.requestId ? goneText + " Reference " + apiErr.requestId + "." : goneText, function () {
        startNewConversation({ skipConfirm: true, skipDelete: true, keepInput: true });
      }, COPY.startNew);
      return;
    }

    if (apiErr.status === 403 && !turn.csrfRetried) {
      // The CSRF token may be stale: refresh it once, then resend with the same client_turn_id.
      turn.csrfRetried = true;
      setBusy(true, COPY.typing);
      refreshSession()
        .then(function () {
          if (turn.epoch !== app.epoch) {
            return;
          }
          setBusy(false);
          submitTurn(turn);
        })
        .catch(function (refreshErr) {
          if (turn.epoch !== app.epoch) {
            return;
          }
          setBusy(false);
          handleTurnError(turn, refreshErr);
        });
      return;
    }

    if (isRetryable(apiErr)) {
      app.pending = turn; // retry reuses turn.clientTurnId so the server can dedupe
      if (UNKNOWN_OUTCOME_STATUS[apiErr.status] === true) {
        turn.uncertain = true; // the response was lost, not refused: never claim "Not sent"
      }
      setBubbleState(turn.bubble, turn.uncertain ? "unconfirmed" : "failed");
      renderEmailActions();
      updateControls();
      showError(errorText(apiErr), retryPending, COPY.tryAgain);
      return;
    }

    // Not retryable as sent (validation, too large, forbidden, reused id): give the text back.
    dropTurn(turn, true);
    showError(errorText(apiErr));
    if (turn.uncertain) {
      resyncNow(); // an earlier attempt may have been processed: show what the server has
    }
  }

  // ------------------------------------------------------------------ wiring

  el.composer.addEventListener("submit", function (event) {
    event.preventDefault();
    sendText();
  });

  el.input.addEventListener("keydown", function (event) {
    if (event.key !== "Enter" || event.shiftKey || event.isComposing || event.keyCode === 229) {
      return;
    }
    event.preventDefault();
    sendText();
  });

  el.input.addEventListener("input", updateControls);

  el.newConversation.addEventListener("click", function () {
    startNewConversation();
  });

  el.errorRetry.addEventListener("click", function () {
    var action = app.retryAction;
    if (action) {
      hideError();
      action();
    }
  });

  el.errorDismiss.addEventListener("click", function () {
    hideError();
    dismissPending();
    focusInput();
  });

  watchScroller();
  applyState(INITIAL_STATE);
  boot();
})();
