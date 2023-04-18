# ! NO LONGER MAINTAINED !

> **Warning**
> This repository is no longer maintained. Please see other forks,
> such as [wfox4/WebChatRWKVv2](https://github.com/wfox4/WebChatRWKVv2). For
> more info on the RWKV language model, check the
> [BlinkDL/ChatRWKV](https://github.com/BlinkDL/ChatRWKV) repository.

---

# WebChatRWKVstic

![screenshot](https://raw.githubusercontent.com/hizkifw/WebChatRWKVstic/main/.github/images/screenshot.png)

[RWKV-V4](https://github.com/BlinkDL/RWKV-LM) inference via
[rwkvstic](https://github.com/harrisonvanderbyl/rwkvstic), with a ChatGPT-like
web UI, including real-time response streaming.

## How to use

```sh
# Clone this repository
git clone https://github.com/hizkifw/WebChatRWKVstic.git
cd WebChatRWKVstic

# Recommended: set up a virtual environment
python -m venv venv
source ./venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run the webserver
python main.py
```

The script will automatically download a suitable RWKV model into the `models`
folder. If you already have a model, you can create the `models` directory and
place your `.pth` file there.

## Currently state

- Mobile-friendly web UI with autoscroll, response streaming, markdown
  formatting, and syntax highlighting
- Input is formatted into a question/answer format for the model, and earlier
  chat messages are included in the context

## TODO

- Tune the model to better match ChatGPT
- Clean up the code
- Create a Docker image
