(() => {
  const messages = {};
  let isReady = false;

  // Wait until the whole page has loaded
  window.addEventListener("load", () => {
    const chatbox = document.querySelector("#chatbox");
    const chatform = document.querySelector("#chatform");
    const historybox = document.querySelector("#history");

    marked.setOptions({
      highlight: function (code, lang) {
        const language = hljs.getLanguage(lang) ? lang : "plaintext";
        return hljs.highlight(code, { language }).value;
      },
      langPrefix: "hljs language-",
    });

    const renderMessage = (id, from, message) => {
      messages[id] = message;

      const div = document.createElement("div");
      div.id = id;
      div.className = "message";

      const tname = document.createElement("h4");
      tname.innerText = from;
      div.appendChild(tname);

      const txt = document.createElement("div");
      txt.innerHTML = marked.parse(message);
      txt.className = "messagecontent";
      div.appendChild(txt);

      historybox.appendChild(div);
    };

    const appendMessage = (id, message) => {
      messages[id] += message;

      let markdown = messages[id];
      // Check for open code blocks and close them
      if ((markdown.match(/```/g) || []).length % 2 !== 0) markdown += "\n```";

      // Append to the p
      const p = document.querySelector("#" + id + " > .messagecontent");
      p.innerHTML = marked.parse(markdown);

      // Scroll the history box
      historybox.scrollTo({
        behavior: "smooth",
        top: historybox.scrollHeight,
        left: 0,
      });
    };

    const makeId = () =>
      (Date.now().toString(36) + Math.random().toString(36)).replace(".", "");

    // Connect to websocket
    const ws = new WebSocket(
      location.protocol.replace("http", "ws") + "//" + location.host + "/ws"
    );

    // Attach event listener
    ws.addEventListener("message", (ev) => {
      data = JSON.parse(ev.data);
      if ("result" in data && "token" in data["result"]) {
        if (data.result.token === null) isReady = true;
        else appendMessage(data.id, data.result.token.replace("<", "&lt;"));
      }
    });
    ws.addEventListener("open", () => {
      isReady = true;
      renderMessage(makeId(), "[system]", "WebSocket connected!");
    });
    ws.addEventListener("close", () => {
      isReady = false;
      renderMessage(makeId(), "[system]", "WebSocket disconnected!");
    });

    const sendMessage = async (message) => {
      isReady = false;

      // Generate an ID for the response
      respid = makeId();

      // Add message to the page
      renderMessage(makeId(), "User", message);
      renderMessage(respid, "ChatRWKV", "");

      // Send message to server
      ws.send(
        JSON.stringify({
          jsonrpc: "2.0",
          method: "chat",
          params: {
            text: message,
          },
          id: respid,
        })
      );
    };

    const onSubmit = () => {
      if (!isReady) return;
      sendMessage(chatbox.value.trim());
      chatbox.value = "";
    };

    chatform.addEventListener("submit", (e) => {
      e.preventDefault();
      onSubmit();
    });
    chatbox.addEventListener("keydown", (e) => {
      if (e.key == "Enter" && !e.shiftKey) onSubmit();
    });
  });
})();
