import asyncio

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

import model

app = FastAPI()


@app.websocket("/ws")
async def websocket(ws: WebSocket):
    loop = asyncio.get_running_loop()
    await ws.accept()

    session = {"state": None}

    async def reply(id, *, result=None, error=None):
        either = (result is None) is not (error is None)
        assert either, "Either result or error must be set!"

        if result is not None:
            await ws.send_json({"jsonrpc": "2.0", "result": result, "id": id})
        elif error is not None:
            await ws.send_json({"jsonrpc": "2.0", "error": error, "id": id})

    def on_progress(id):
        def callback(res):
            asyncio.run_coroutine_threadsafe(reply(id, result={"token": res}), loop)

        return callback

    def on_done(input):
        def callback(result):
            print("--- input ---")
            print(input)
            print("--- output ---")
            print(result["output"])
            print("---")

            session["state"] = result["state"]

        return callback

    while True:
        data = await ws.receive_json()
        if "jsonrpc" not in data or data["jsonrpc"] != "2.0":
            await reply(
                data.get("id", None) if type(data) == dict else None,
                error="invalid message",
            )

        method, params, id = (
            data.get("method", None),
            data.get("params", None),
            data.get("id", None),
        )

        if method == "chat":
            text = params.get("text", None)
            if text is None:
                await reply(id, error="text is required")

            await loop.run_in_executor(
                None,
                model.chat,
                session["state"],
                text,
                on_progress(id),
                on_done(text),
            )
        else:
            await reply(id, error=f"invalid method '{method}'")


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
