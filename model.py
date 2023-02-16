import queue
import threading
import traceback

import torch
from rwkvstic.agnostic.backends import TORCH
from rwkvstic.load import RWKV


def no_tqdm():
    from functools import partialmethod

    from tqdm import tqdm

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


# Load the model (supports full path, relative path, and remote paths)
model = RWKV(
    "../ChatRWKV/models/RWKV-4-Pile-169M-20220807-8023.pth",
    # "../ChatRWKV/models/RWKV-4-Pile-7B-Instruct-test2-20230209.pth",
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


def inferthread():
    while True:
        try:
            # Get task
            task = inferqueue.get()

            print("Infer:", task["context"])

            # Perform inference
            model.setState(task.get("state", model.emptyState))
            model.loadContext(newctx=task["context"])
            res = model.forward(
                number=100,
                temp=0.7,
                top_p_usual=0.9,
                progressLambda=task["progress_callback"],
            )

            if "done_callback" in task:
                task["done_callback"](res)

            print(" ->", res["output"])
        except Exception:
            traceback.print_exc()
        finally:
            task["progress_callback"](None)


t = threading.Thread(target=inferthread, daemon=True)
t.start()


def infer(*, context: str, state=None, on_progress=None, on_done=None):
    def _progress_callback(args):
        if on_progress is None:
            return

        if args is None:
            on_progress(None)
            return

        last_token = args["tokens"][-1]
        on_progress(model.tokenizer.decode(last_token))

    def _done_callback(result):
        if on_done is None:
            return
        on_done(result)

    task = {
        "context": context,
        "state": state if state is not None else model.emptyState,
        "progress_callback": _progress_callback,
        "done_callback": _done_callback,
    }
    inferqueue.put(task)


def chat(state, input: str, on_progress, on_done):
    if state is None:
        state = model.emptyState

    infer(
        context=input,
        state=state,
        on_progress=on_progress,
        on_done=on_done,
    )


if __name__ == "__main__":
    while True:
        print("")
        line_in = input("> ").replace("\\n", "\n").strip()
        if line_in == "/reset":
            print(model.getState())
            model.resetState()
            print("State has been reset.")
            continue

        model.loadContext(newctx=line_in)

        def on_progress(args):
            last_token = args["tokens"][-1]
            print(model.tokenizer.decode(last_token), end="")

        output = model.forward(
            number=100,
            temp=0.7,
            top_p_usual=0.9,
            progressLambda=on_progress,
            stopStrings=["<|endoftext|>"],
        )
