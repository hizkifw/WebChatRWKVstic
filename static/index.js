(() => {
  // Wait until the whole page has loaded
  window.addEventListener("load", () => {
    const chatbox = document.querySelector("#chatbox");
    const chatform = document.querySelector("#chatform");
    const historybox = document.querySelector("#history");

    const renderMessage = (id, from, message) => {
      const div = document.createElement("div");
      div.id = id;

      const tname = document.createElement("h4");
      tname.innerText = from;
      div.appendChild(tname);

      const txt = document.createElement("p");
      txt.innerText = message;
      div.appendChild(txt);

      historybox.appendChild(div);
    };

    const appendMessage = (id, message) => {
      // Append to the p
      document.querySelector("#" + id + ">p").innerText += message;
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
        appendMessage(data.id, data.result.token);
      }
    });
    ws.addEventListener("open", () => {
      renderMessage(makeId(), "[system]", "WebSocket connected!");
    });
    ws.addEventListener("close", () => {
      renderMessage(makeId(), "[system]", "WebSocket disconnected!");
    });

    const sendMessage = async (message) => {
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
      sendMessage(chatbox.value);
      chatbox.value = "";
    });
  });
})();
