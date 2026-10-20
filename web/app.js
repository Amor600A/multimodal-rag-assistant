/* 多模态 RAG 助手 —— 前端逻辑
 * 约束：所有动态内容一律使用 textContent / createElement 渲染，禁止 innerHTML。
 * SSE：用 fetch + ReadableStream 手动解析 data: 帧（EventSource 不支持 POST）。
 */

(function () {
  "use strict";

  var API = "/api";
  var K_ACCESS = "rag_access_token";
  var K_REFRESH = "rag_refresh_token";
  var K_USER = "rag_user";

  // ---------- 状态 ----------
  var state = {
    currentThreadId: null,
    sending: false,
    pendingImageId: null,
    pendingImageUrl: null,
    sessions: [],
  };

  // ---------- DOM ----------
  var $ = function (id) { return document.getElementById(id); };

  var loginView = $("login-view");
  var appView = $("app-view");
  var loginForm = $("login-form");
  var loginBtn = $("login-btn");
  var loginError = $("login-error");
  var userInfoEl = $("user-info");
  var sessionListEl = $("session-list");
  var messagesEl = $("messages");
  var emptyTip = $("empty-tip");
  var inputEl = $("input");
  var sendBtn = $("send-btn");
  var imageInput = $("image-input");
  var imagePreview = $("image-preview");
  var sidebar = $("sidebar");

  // ---------- token 存取 ----------
  function getAccess() { return localStorage.getItem(K_ACCESS); }
  function getRefresh() { return localStorage.getItem(K_REFRESH); }
  function setTokens(access, refresh) {
    if (access) localStorage.setItem(K_ACCESS, access);
    if (refresh) localStorage.setItem(K_REFRESH, refresh);
  }
  function clearAuth() {
    localStorage.removeItem(K_ACCESS);
    localStorage.removeItem(K_REFRESH);
    localStorage.removeItem(K_USER);
  }
  function getUser() {
    try { return JSON.parse(localStorage.getItem(K_USER) || "null"); }
    catch (e) { return null; }
  }

  // ---------- 网络 ----------
  // 401 时自动 refresh 一次并重试
  async function apiFetch(path, options) {
    options = options || {};
    options.headers = options.headers || {};
    var access = getAccess();
    if (access) options.headers["Authorization"] = "Bearer " + access;

    var res = await fetch(API + path, options);
    if (res.status !== 401) return res;

    var refresh = getRefresh();
    if (!refresh) return res;

    var r = await fetch(API + "/auth/refresh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refresh }),
    });
    if (!r.ok) {
      clearAuth();
      return res;
    }
    var data = await r.json();
    setTokens(data.access_token, data.refresh_token);

    options.headers["Authorization"] = "Bearer " + data.access_token;
    return fetch(API + path, options);
  }

  async function readError(res) {
    try {
      var d = await res.json();
      if (d && d.detail) return String(d.detail);
    } catch (e) { /* 忽略 */ }
    return "请求失败（" + res.status + "）";
  }

  // ---------- 登录 / 登出 ----------
  async function doLogin(username, password) {
    loginBtn.disabled = true;
    loginError.textContent = "";
    try {
      var res = await fetch(API + "/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: username, password: password }),
      });
      if (!res.ok) {
        loginError.textContent = (res.status === 429)
          ? "尝试过于频繁，请 1 分钟后再试"
          : await readError(res);
        return;
      }
      var data = await res.json();
      setTokens(data.access_token, data.refresh_token);
      localStorage.setItem(K_USER, JSON.stringify(data.user || null));
      enterApp(data.user);
      await loadSessions();
    } catch (e) {
      loginError.textContent = "网络错误：" + e.message;
    } finally {
      loginBtn.disabled = false;
    }
  }

  async function doLogout() {
    try {
      await apiFetch("/auth/logout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
    } catch (e) { /* 忽略 */ }
    clearAuth();
    state.currentThreadId = null;
    state.sessions = [];
    appView.classList.add("hidden");
    loginView.classList.remove("hidden");
    inputEl.value = "";
    messagesEl.replaceChildren(emptyTip);
  }

  function enterApp(user) {
    loginView.classList.add("hidden");
    appView.classList.remove("hidden");
    if (user) {
      var label = user.username || "";
      if (user.dept_name) label += " · " + user.dept_name;
      if (user.role === "admin") label += " · 管理员";
      userInfoEl.textContent = label;
    }
  }

  // ---------- 会话列表 ----------
  function renderSessions() {
    sessionListEl.replaceChildren();
    state.sessions.forEach(function (s) {
      var li = document.createElement("li");
      li.className = "session-item" + (s.thread_id === state.currentThreadId ? " active" : "");

      var name = document.createElement("span");
      name.className = "session-name";
      name.textContent = s.title || s.thread_id;
      name.title = s.title || s.thread_id;

      var del = document.createElement("button");
      del.className = "session-del";
      del.type = "button";
      del.textContent = "删除";
      del.addEventListener("click", function (ev) {
        ev.stopPropagation();
        deleteSession(s.thread_id);
      });

      li.appendChild(name);
      li.appendChild(del);
      li.addEventListener("click", function () { selectSession(s.thread_id); });
      sessionListEl.appendChild(li);
    });
  }

  async function loadSessions() {
    try {
      var res = await apiFetch("/sessions");
      if (res.status === 401) { doLogout(); return; }
      if (!res.ok) return;
      state.sessions = await res.json();
      renderSessions();
    } catch (e) { /* 忽略 */ }
  }

  async function deleteSession(threadId) {
    try {
      var res = await apiFetch("/sessions/" + encodeURIComponent(threadId), { method: "DELETE" });
      if (res.status === 401) { doLogout(); return; }
      if (res.ok || res.status === 204) {
        state.sessions = state.sessions.filter(function (s) { return s.thread_id !== threadId; });
        if (state.currentThreadId === threadId) newSession();
        renderSessions();
      }
    } catch (e) { /* 忽略 */ }
  }

  function selectSession(threadId) {
    // 后端未提供历史消息接口，切换会话仅切换上下文并清空视图
    state.currentThreadId = threadId;
    messagesEl.replaceChildren(emptyTip);
    renderSessions();
    closeSidebarOnNarrow();
  }

  function newSession() {
    state.currentThreadId = null;
    messagesEl.replaceChildren(emptyTip);
    renderSessions();
    clearPendingImage();
    closeSidebarOnNarrow();
    inputEl.focus();
  }

  function closeSidebarOnNarrow() {
    if (window.innerWidth <= 768) sidebar.classList.remove("open");
  }

  // ---------- 消息渲染 ----------
  function removeEmptyTip() {
    if (emptyTip.parentNode === messagesEl) messagesEl.removeChild(emptyTip);
  }

  function appendUserMsg(text, imgUrl) {
    removeEmptyTip();
    var wrap = document.createElement("div");
    wrap.className = "msg user";
    var bubble = document.createElement("div");
    bubble.className = "bubble";

    if (imgUrl) {
      var img = document.createElement("img");
      img.className = "thumb";
      img.src = imgUrl;
      img.alt = "上传图片";
      bubble.appendChild(img);
    }
    if (text) bubble.appendChild(document.createTextNode(text));

    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    scrollToBottom();
  }

  function createAssistantBubble() {
    removeEmptyTip();
    var wrap = document.createElement("div");
    wrap.className = "msg assistant";
    var bubble = document.createElement("div");
    bubble.className = "bubble";

    var textEl = document.createElement("span");
    textEl.className = "text";
    bubble.appendChild(textEl);
    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    scrollToBottom();
    return { wrap: wrap, bubble: bubble, textEl: textEl };
  }

  function appendWarning(detail) {
    var bar = document.createElement("div");
    bar.className = "warn-bar";
    bar.textContent = "提示：" + detail;
    messagesEl.appendChild(bar);
    scrollToBottom();
  }

  function renderSources(bubbleEl, textEl, fullText) {
    // 末尾 "来源: [1]" 单独渲染为引用行
    var lines = fullText.split("\n");
    var idx = -1;
    for (var i = lines.length - 1; i >= 0; i--) {
      if (lines[i].trim().indexOf("来源:") === 0) { idx = i; break; }
    }
    if (idx < 0) return false;

    var body = lines.slice(0, idx).join("\n");
    var src = lines.slice(idx).join("\n").trim();

    textEl.textContent = body;
    var srcSpan = document.createElement("span");
    srcSpan.className = "src-line";
    srcSpan.textContent = src;
    bubbleEl.appendChild(srcSpan);
    return true;
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  // ---------- 图片上传 ----------
  function clearPendingImage() {
    if (state.pendingImageUrl) URL.revokeObjectURL(state.pendingImageUrl);
    state.pendingImageId = null;
    state.pendingImageUrl = null;
    imagePreview.replaceChildren();
    imagePreview.classList.add("hidden");
    imageInput.value = "";
  }

  async function handleImagePicked(file) {
    clearPendingImage();
    if (!file) return;

    // 本地预览
    state.pendingImageUrl = URL.createObjectURL(file);
    imagePreview.classList.remove("hidden");
    var img = document.createElement("img");
    img.src = state.pendingImageUrl;
    img.alt = "预览";
    var stateText = document.createElement("span");
    stateText.className = "upload-state";
    stateText.textContent = "上传中…";
    var removeBtn = document.createElement("button");
    removeBtn.className = "remove-img";
    removeBtn.type = "button";
    removeBtn.textContent = "移除";
    removeBtn.addEventListener("click", clearPendingImage);
    imagePreview.replaceChildren(img, stateText, removeBtn);

    var fd = new FormData();
    fd.append("file", file);
    try {
      var res = await apiFetch("/knowledge/images", { method: "POST", body: fd });
      if (res.status === 401) { doLogout(); return; }
      if (!res.ok) {
        stateText.textContent = await readError(res);
        return;
      }
      var data = await res.json();
      state.pendingImageId = data.image_id;
      stateText.textContent = "已上传，待发送";
    } catch (e) {
      stateText.textContent = "上传失败：" + e.message;
    }
  }

  // ---------- SSE 流式发送 ----------
  var NODE_LABEL = {
    retrieve: "检索知识库…",
    generate: "生成回答…",
    output_guard: "校验引用…",
  };

  async function ensureThread(firstText) {
    if (state.currentThreadId) return state.currentThreadId;
    var title = firstText.replace(/\s+/g, " ").trim().slice(0, 20) || "新会话";
    var res = await apiFetch("/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: title }),
    });
    if (!res.ok) return null;
    var s = await res.json();
    state.currentThreadId = s.thread_id;
    state.sessions.unshift({ thread_id: s.thread_id, title: s.title, created_at: s.created_at });
    renderSessions();
    return s.thread_id;
  }

  function lockInput() {
    state.sending = true;
    sendBtn.disabled = true;
    inputEl.disabled = true;
  }
  function unlockInput() {
    state.sending = false;
    sendBtn.disabled = false;
    inputEl.disabled = false;
    inputEl.focus();
  }

  // 建立 chat SSE 连接；遇 401 时自动刷新 token 后重试一次。
  // 返回 Response；鉴权彻底失败（已登出）时返回 null，调用方需停止后续处理。
  async function openChatStream(message, threadId, imageId) {
    var body = JSON.stringify({ message: message, thread_id: threadId, image_id: imageId });
    var res = await fetch(API + "/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + getAccess(),
      },
      body: body,
    });
    if (res.status !== 401) return res;

    var refresh = getRefresh();
    if (!refresh) {
      doLogout();
      return null;
    }
    var r = await fetch(API + "/auth/refresh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refresh }),
    });
    if (!r.ok) {
      clearAuth();
      doLogout();
      return null;
    }
    var data = await r.json();
    setTokens(data.access_token, data.refresh_token);
    return fetch(API + "/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + data.access_token,
      },
      body: body,
    });
  }

  async function send() {
    if (state.sending) return;

    var text = inputEl.value.trim();
    if (!text && !state.pendingImageId) return;

    var imageId = state.pendingImageId;
    var imageUrl = state.pendingImageUrl;
    var payloadText = text || "[图片]";

    var threadId = await ensureThread(payloadText);
    if (!threadId) {
      appendWarning("会话创建失败，请稍后重试");
      return;
    }

    // 上屏用户消息
    appendUserMsg(text, imageUrl);
    inputEl.value = "";
    autoGrow(inputEl);
    // 图片预览转为气泡内缩略图后释放预览区
    state.pendingImageId = null;
    state.pendingImageUrl = null;
    imagePreview.replaceChildren();
    imagePreview.classList.add("hidden");
    imageInput.value = "";

    var ref = createAssistantBubble();
    lockInput();

    var gotAnyToken = false;
    var finished = false;
    var fullText = "";

    try {
      // 逐次尝试：仅当"连接建立阶段失败且尚未收到任何 token"时自动重试 1 次
      // （含 401 刷新后重试）。一旦收到 token 就绝不重试——LLM 生成不幂等。
      for (var attempt = 1; attempt <= 2; attempt++) {
        gotAnyToken = false;
        finished = false;
        fullText = "";
        ref.textEl.textContent = "";
        ref.bubble.replaceChildren(ref.textEl);

        try {
          var res = await openChatStream(payloadText, threadId, imageId);
          if (res === null) return;   // 鉴权彻底失败，已登出
          if (!res.ok) {
            ref.textEl.textContent = "发送失败：" + (await readError(res));
            return;
          }

          var reader = res.body.getReader();
          var decoder = new TextDecoder("utf-8");
          var buffer = "";

          while (true) {
            var chunk = await reader.read();
            if (chunk.done) break;
            buffer += decoder.decode(chunk.value, { stream: true });
            // SSE 规范允许 CRLF/LF/CR 三种换行；sse_starlette 默认用 CRLF，
            // 统一归一化为 LF 后再按空行切帧，避免跨块时把已分割的 \r\n 误判为帧边界。
            buffer = buffer.replace(/\r\n/g, "\n");

            var sep;
            while ((sep = buffer.indexOf("\n\n")) >= 0) {
              var rawFrame = buffer.slice(0, sep);
              buffer = buffer.slice(sep + 2);
              var parsed = parseFrame(rawFrame);
              if (!parsed) continue;

              if (parsed.type === "token") {
                gotAnyToken = true;
                fullText += parsed.content || "";
                ref.textEl.textContent = fullText;
                scrollToBottom();
              } else if (parsed.type === "node") {
                var label = NODE_LABEL[parsed.node];
                if (label && !gotAnyToken) ref.textEl.textContent = label;
              } else if (parsed.type === "warning") {
                appendWarning(parsed.detail || "输出已被净化");
              } else if (parsed.type === "done") {
                finished = true;
                renderSources(ref.bubble, ref.textEl, fullText);
              } else if (parsed.type === "error") {
                finished = true;
                if (!fullText) {
                  ref.textEl.textContent = "出错了：" + (parsed.detail || "未知错误");
                } else {
                  ref.textEl.textContent = fullText;
                  renderSources(ref.bubble, ref.textEl, fullText);
                  appendWarning("生成中断：" + (parsed.detail || "未知错误"));
                }
              }
            }
          }

          if (finished) return;

          // 流结束但无终态帧
          if (gotAnyToken) {
            // 已产生内容：保留文本并提示重发，不自动重试
            renderSources(ref.bubble, ref.textEl, fullText);
            appendWarning("连接中断，回答可能不完整，请重新发送");
            return;
          }
          // 未收到任何 token → 属连接建立阶段失败，可重试
          if (attempt < 2) continue;
          ref.textEl.textContent = "连接中断，请重新发送";
          return;
        } catch (e) {
          if (!gotAnyToken && attempt < 2) continue;
          if (gotAnyToken) {
            renderSources(ref.bubble, ref.textEl, fullText);
            appendWarning("连接中断，回答可能不完整，请重新发送");
          } else {
            ref.textEl.textContent = "网络错误：" + e.message;
          }
          return;
        }
      }
    } finally {
      unlockInput();
    }
  }

  // 解析单帧：收集所有 data: 行并 JSON.parse
  function parseFrame(rawFrame) {
    var lines = rawFrame.split("\n");
    var dataLines = [];
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];
      if (line.indexOf("data:") === 0) {
        dataLines.push(line.slice(5).replace(/^ /, ""));
      }
    }
    if (dataLines.length === 0) return null;
    try {
      return JSON.parse(dataLines.join("\n"));
    } catch (e) {
      return null;
    }
  }

  // ---------- 输入框自适应高度 ----------
  function autoGrow(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  }

  // ---------- 事件绑定 ----------
  loginForm.addEventListener("submit", function (ev) {
    ev.preventDefault();
    doLogin($("login-username").value.trim(), $("login-password").value);
  });

  $("logout-btn").addEventListener("click", doLogout);
  $("new-session-btn").addEventListener("click", newSession);
  $("sidebar-toggle").addEventListener("click", function () {
    sidebar.classList.toggle("open");
  });

  sendBtn.addEventListener("click", send);

  inputEl.addEventListener("input", function () { autoGrow(inputEl); });
  inputEl.addEventListener("keydown", function (ev) {
    if (ev.key === "Enter" && !ev.shiftKey) {
      ev.preventDefault();
      if (!sendBtn.disabled) send();
    }
  });

  imageInput.addEventListener("change", function (ev) {
    handleImagePicked(ev.target.files && ev.target.files[0]);
  });

  // ---------- 启动 ----------
  (async function boot() {
    var user = getUser();
    if (!getAccess()) return;
    // 校验 token 有效性并补全用户信息
    try {
      var res = await apiFetch("/auth/me");
      if (res.ok) {
        var me = await res.json();
        localStorage.setItem(K_USER, JSON.stringify(me));
        enterApp(me);
        await loadSessions();
      } else {
        clearAuth();
      }
    } catch (e) {
      clearAuth();
    }
  })();
})();
