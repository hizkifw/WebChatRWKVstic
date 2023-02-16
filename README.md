# WebChatRWKVstic

![screenshot](https://raw.githubusercontent.com/hizkifw/WebChatRWKVstic/main/.github/images/screenshot.png)

[RWKV-V4](https://github.com/BlinkDL/RWKV-LM) inference via
[rwkvstic](https://github.com/harrisonvanderbyl/rwkvstic), with a ChatGPT-like
web UI, including real-time response streaming.

Currently state:

- Mobile-friendly web UI with autoscroll, response streaming, markdown
  formatting, and syntax highlighting
- Chat message is just given as raw input to the model, and each message is
  evaluated separately

TODO:

- Tune the model to better match ChatGPT
- Clean up the code
- Create a Docker image
