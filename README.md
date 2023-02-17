# WebChatRWKVstic

![screenshot](https://raw.githubusercontent.com/hizkifw/WebChatRWKVstic/main/.github/images/screenshot.png)

[RWKV-V4](https://github.com/BlinkDL/RWKV-LM) inference via
[rwkvstic](https://github.com/harrisonvanderbyl/rwkvstic), with a ChatGPT-like
web UI, including real-time response streaming.

## How to use

1. Clone this repo: `git clone https://github.com/hizkifw/WebChatRWKVstic.git`
2. Edit the `model.py` file and change the model path and settings
3. Run it:

```sh
# Recommended: run in a virtual environment
python -m venv venv
source ./venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run the webserver
python main.py
```

## Currently state

- Mobile-friendly web UI with autoscroll, response streaming, markdown
  formatting, and syntax highlighting
- Input is formatted into a question/answer format for the model, and earlier
  chat messages are included in the context

## TODO

- Tune the model to better match ChatGPT
- Clean up the code
- Create a Docker image
