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
    # "../ChatRWKV/models/RWKV-4-Pile-169M-20220807-8023.pth",
    # "../ChatRWKV/models/RWKV-4-Pile-7B-Instruct-test2-20230209.pth",
    "../ChatRWKV/models/RWKV-4-Pile-7B-20230109-ctx4096.pth",
    mode=TORCH,
    useGPU=True,
    runtimedtype=torch.float32,
    dtype=torch.bfloat16,
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
            model.resetState()
            model.loadContext(newctx=task["context"])
            res = model.forward(
                number=100,
                temp=0.7,
                top_p_usual=0.9,
                progressLambda=task["callback"],
            )

            print(" ->", res["output"])
        except Exception:
            traceback.print_exc()
        finally:
            task["callback"](None)


t = threading.Thread(target=inferthread, daemon=True)
t.start()


def infer(context: str, callback):
    def _callback(args):
        if args is None:
            callback(None)
            return

        last_token = args["tokens"][-1]
        callback(model.tokenizer.decode(last_token))

    task = {
        "context": context,
        "callback": _callback,
    }
    inferqueue.put(task)


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
