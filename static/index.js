(() => {
  const messages = {};
  let isReady = false;

  // Wait until the whole page has loaded
  window.addEventListener("load", () => {
    const chatbox = document.querySelector("#chatbox");
    const chatform = document.querySelector("#chatform");
    const historybox = document.querySelector("#history");

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

      // Append to the p
      const p = document.querySelector("#" + id + " > .messagecontent");
      p.innerHTML = marked.parse(messages[id]);

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
        else appendMessage(data.id, data.result.token);
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

    chatform.addEventListener("submit", (e) => {
      e.preventDefault();

      if (!isReady) return;
      sendMessage(chatbox.value);
      chatbox.value = "";
    });
  });
})();
