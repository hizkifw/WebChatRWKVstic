# WebChatRWKVstic

[RWKV-V4](https://github.com/BlinkDL/RWKV-LM) inference via
[rwkvstic](https://github.com/harrisonvanderbyl/rwkvstic), with a ChatGPT-like
web UI, including real-time response streaming.

Currently state:

- Mobile-friendly web UI with autoscroll, response streaming, and markdown
  formatting
- Chat message is just given as raw input to the model, and each message is
  evaluated separately

TODO:

- Keep track of state between chat messages, and handle multiple users
- Multiline chat box input
- Tune the model to better match ChatGPT
