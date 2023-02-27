import queue
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from rwkvstic.agnostic.backends import TORCH
from rwkvstic.load import RWKV


def no_tqdm():
    from functools import partialmethod

    from tqdm import tqdm

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


@dataclass
class OnlineModel:
    name: str
    url: str
    sha256: str
    vram_gb: int


online_models = [
    # TODO: add more models here
    OnlineModel(
        name="RWKV-4-Pile-7B-ctx4096",
        url="https://huggingface.co/BlinkDL/rwkv-4-pile-7b/resolve/main/RWKV-4-Pile-7B-20230109-ctx4096.pth",
        sha256="9ea1271b25deb6c72bd29f629147d5013cc7d7c69f9715192f6b6b92fca08f64",
        vram_gb=14,
    ),
    OnlineModel(
        name="RWKV-4-Pile-3B-ctx4096",
        url="https://huggingface.co/BlinkDL/rwkv-4-pile-3b/resolve/main/RWKV-4-Pile-3B-20221110-ctx4096.pth",
        sha256="9500633f23d86fbae3cb3cbe7908b97b971e9561edf583c2c5c60b10b02bcc27",
        vram_gb=6,
    ),
    OnlineModel(
        name="RWKV-4-Pile-1B5-ctx4096",
        url="https://huggingface.co/BlinkDL/rwkv-4-pile-1b5/resolve/main/RWKV-4-Pile-1B5-20220929-ctx4096.pth",
        sha256="6c97043e1bb0867368249290c97a2fe8ffc5ec12ceb1b5251f4ee911f9982c23",
        vram_gb=3.7,
    ),
    OnlineModel(
        name="RWKV-4-Pile-1B5-Instruct-test2",
        url="https://huggingface.co/BlinkDL/rwkv-4-pile-1b5/resolve/main/RWKV-4-Pile-1B5-Instruct-test2-20230209.pth",
        sha256="19aafd001257702bd66c81e5e05dcbc088341e825cc41b4feaeb35aa1b55624c",
        vram_gb=3.7,
    ),
    OnlineModel(
        name="RWKV-4-Pile-169M",
        url="https://huggingface.co/BlinkDL/rwkv-4-pile-169m/resolve/main/RWKV-4-Pile-169M-20220807-8023.pth",
        sha256="713c6f6137a08d3a86ab57df4f09ea03563329beb3bbabc23509d6c57aa0f9e2",
        vram_gb=1.3,
    ),
]


def hash_file(filename):
    import hashlib

    file_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        while True:
            data = f.read(4 * 1024)
            if not data:
                break
            file_hash.update(data)
    return file_hash.hexdigest()


# https://stackoverflow.com/a/63831344
def download(url, filename, sha256=None):
    import functools
    import pathlib
    import shutil
    import requests
    from tqdm.auto import tqdm

    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get("Content-Length", 0))

    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(
        r.raw.read, decode_content=True
    )  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)

    if sha256 is not None:
        print("Verifying file integrity...")
        file_hash = hash_file(path)
        if file_hash != sha256:
            print("Error downloading file: checksums do not match")
            print("Expected", sha256)
            print("But got ", file_hash)
            raise Exception("Checksums do not match!")

    return path


def get_checkpoint():
    import psutil
    import os
    from glob import glob
    from os import path

    has_cuda = torch.cuda.is_available()
    ram_total = psutil.virtual_memory().total
    vram_total = 0

    # Check if CUDA is available
    if has_cuda:
        print("CUDA available")
        vram_total = torch.cuda.mem_get_info()[1]
    else:
        print(
            """
**************************************
WARN: CUDA not available, will use CPU
If you want to use CUDA, try running this command:

  pip install torch --extra-index-url https://download.pytorch.org/whl/cu117 --upgrade

For more information, see: https://pytorch.org/get-started/locally/
*************************************
"""
        )

    models_dir = "models"
    if not path.exists(models_dir):
        os.makedirs(models_dir)

    # Check if there are any models in the models/ folder
    models = glob(path.join(models_dir, "*.pth"))

    if len(models) == 0:
        print("No *.pth models found in the `models` folder, downloading...")
        print(" -> RAM:", ram_total)
        print(" -> VRAM:", vram_total)
        memtarget = vram_total if has_cuda else ram_total
        for m in online_models:
            if m.vram_gb * 1024 * 1024 * 1024 <= memtarget:
                print("Downloading model", m.name)
                download(
                    m.url,
                    path.join(models_dir, m.name + ".pth"),
                    sha256=m.sha256,
                )
                break

        models = glob(path.join(models_dir, "*.pth"))
        if len(models) == 0:
            raise Exception("Could not find a suitable model to download.")

    # TODO: get model name from command line args / config file
    print("-> Using model", models[0])
    return models[0]


# Load the model (supports full path, relative path, and remote paths)
model = RWKV(
    get_checkpoint(),
    mode=TORCH,
    useGPU=torch.cuda.is_available(),
    runtimedtype=torch.float32,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)

# Disable tqdm
no_tqdm()


@dataclass
class Task:
    state: Any = model.emptyState
    context: str = ""
    progress_callback: Callable[[str], None] = lambda x: None
    done_callback: Callable[[dict[str, Any]], None] = lambda x: None
    forward_kwargs: dict = field(default_factory=dict)


inferqueue: queue.Queue[Task] = queue.Queue()


def inferthread():
    while True:
        try:
            # Get task
            task = inferqueue.get()

            # Perform inference
            model.setState(task.state)
            model.loadContext(newctx=task.context)
            res = model.forward(
                number=512,
                temp=1,
                top_p_usual=0.7,
                end_adj=-2,
                progressLambda=task.progress_callback,
                **task.forward_kwargs,
            )

            task.done_callback(res)
        except Exception:
            traceback.print_exc()
        finally:
            task.progress_callback(None)


def infer(
    *,
    context: str,
    state=None,
    on_progress=None,
    on_done=None,
    forward_kwargs={},
):
    ev = threading.Event()

    # args['logits', 'state', 'output', 'progress', 'tokens', 'total', 'current']
    def _progress_callback(args):
        if on_progress is None:
            return

        if args is None:
            on_progress(None, None)
            return

        last_token = args["tokens"][-1]
        token_str = model.tokenizer.decode(last_token)

        on_progress(token_str, args["state"])

    def _done_callback(result):
        ev.set()
        if on_done is None:
            return
        on_done(result)

    task = Task(
        state=state if state is not None else model.emptyState,
        context=context,
        progress_callback=_progress_callback,
        done_callback=_done_callback,
        forward_kwargs=forward_kwargs,
    )
    inferqueue.put(task)
    ev.wait()


print("Loading context")
chat_initial_context = open("prompt.txt").read().strip()
model.loadContext(
    newctx=chat_initial_context,
    progressCallBack=lambda p: print(model.tokenizer.decode(p[-1]), end=""),
)
chat_initial_state = model.getState()
model.resetState()
print("Chat context loaded")

t = threading.Thread(target=inferthread, daemon=True)
t.start()


def chat(state, input: str, on_progress, on_done):
    # Format the input to be a Q & A
    input = f"""
Question:
{input}

Full Answer in Markdown:
"""

    # Set empty state if not provided
    if state is None:
        state = chat_initial_state

    ctx = {"buf": "", "buf_state": None}
    stop_sequences = ["\nQuestion:", "\n---"]

    def _on_progress(token: str, state=None):
        print("token", repr(token))
        if token is None:
            on_progress(None)
            return

        # This chunk of code will look for stop sequences. If found, all text
        # will be stored in the `buf` until either the whole stop sequence is
        # matched, in which case all subsequent progress is dropped, or the
        # sequence doesn't match fully, in which case the buffer will be flushed
        # to the callback.
        #
        # The model state is also stored in the `buf_state`, only when the stop
        # sequences do not match. This allows us to restore the model to right
        # before the stop sequence was produced.
        for ss in stop_sequences:
            if ss == ctx["buf"]:
                return

            if ss.startswith(ctx["buf"] + token):
                ctx["buf"] += token
                if ss == ctx["buf"]:
                    on_progress(None)
                return

        for ss in stop_sequences:
            if ss.startswith(token):
                if len(ctx["buf"]) > 0:
                    on_progress(ctx["buf"])
                ctx["buf"] = token
                if ss == ctx["buf"]:
                    on_progress(None)
                return

        if len(ctx["buf"]) > 0:
            on_progress(ctx["buf"])
            ctx["buf"] = ""

        ctx["buf_state"] = state
        on_progress(token)

    def _on_done(result):
        result["state"] = ctx["buf_state"]
        on_done(result)

    infer(
        context=input,
        state=state,
        on_progress=_on_progress,
        on_done=_on_done,
        forward_kwargs={
            "stopStrings": [
                "<|endoftext|>",
                "---",
                "Question:",
                "Full Answer in Markdown:",
            ]
        },
    )


if __name__ == "__main__":
    session = {"state": None}

    while True:
        print("")
        line_in = input("You> ").replace("\\n", "\n").strip()
        if line_in == "/reset":
            session["state"] = None
            print("State has been reset.")
            continue

        def on_progress(result):
            if result is None:
                print("")
                return
            print(result, end="")

        def on_done(result):
            session["state"] = result["state"]

        print("Bot> ", end="")
        chat(session["state"], line_in, on_progress, on_done)
