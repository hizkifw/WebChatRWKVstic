import queue
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable

import torch
from rwkvstic.agnostic.backends import TORCH
from rwkvstic.load import RWKV


def no_tqdm():
    from functools import partialmethod

    from tqdm import tqdm

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


# Load the model (supports full path, relative path, and remote paths)
model = RWKV(
    # "../ChatRWKV/models/RWKV-4-Pile-169M-20220807-8023.pth",
    "../ChatRWKV/models/RWKV-4-Pile-7B-Instruct-test2-20230209.pth",
    # "../ChatRWKV/models/RWKV-4-Pile-7B-20230109-ctx4096.pth",
    # "../ChatRWKV/models/RWKV-4-Pile-14B-20230213-8019.pth",
    mode=TORCH,
    useGPU=torch.cuda.is_available(),
    runtimedtype=torch.float32,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)

# Disable tqdm
no_tqdm()

inferqueue = queue.Queue()


@dataclass
class Task:
    state: Any
    context: str
    progress_callback: Callable[[str], None]


def inferthread():
    while True:
        try:
            # Get task
            task = inferqueue.get()

            # Perform inference
            model.setState(task.get("state", model.emptyState))
            model.loadContext(newctx=task["context"])
            res = model.forward(
                number=512,
                temp=0.7,
                top_p_usual=1,
                progressLambda=task["progress_callback"],
                **task.get("forward_kwargs", {}),
            )

            if "done_callback" in task:
                task["done_callback"](res)
        except Exception:
            traceback.print_exc()
        finally:
            task["progress_callback"](None)


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

    task = {
        "context": context,
        "state": state if state is not None else model.emptyState,
        "progress_callback": _progress_callback,
        "done_callback": _done_callback,
        "forward_kwargs": forward_kwargs,
    }
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
    input_indent = "    " + input.replace("\n", "\n    ").strip()
    input = f"""
Question:
{input_indent}

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
