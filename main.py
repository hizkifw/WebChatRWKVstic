import asyncio

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

import model

app = FastAPI()


@app.websocket("/ws")
async def websocket(ws: WebSocket):
    await ws.accept()

    loop = asyncio.get_running_loop()

    def make_callback(id):
        def callback(res):
            asyncio.run_coroutine_threadsafe(
                ws.send_json(
                    {
                        "jsonrpc": "2.0",
                        "result": {"token": res},
                        "id": id,
                    }
                ),
                loop,
            )

        return callback

    while True:
        data = await ws.receive_json()
        if "jsonrpc" not in data or data["jsonrpc"] != "2.0":
            await ws.send_json(
                {
                    "jsonrpc": "2.0",
                    "error": "invalid message",
                    "id": data.get("id", None) if type(data) == dict else None,
                }
            )

        method, params, id = (
            data.get("method", None),
            data.get("params", None),
            data.get("id", None),
        )

        if method == "chat":
            text = params.get("text", None)
            if text is None:
                await ws.send_json(
                    {
                        "jsonrpc": "2.0",
                        "error": f"text is required",
                        "id": id,
                    }
                )

            await loop.run_in_executor(
                None,
                model.infer,
                text,
                make_callback(id),
            )
        else:
            await ws.send_json(
                {
                    "jsonrpc": "2.0",
                    "error": f"invalid method '{method}'",
                    "id": id,
                }
            )


@app.post("/predict")
async def predict():
    pass


app.mount("/", StaticFiles(directory="static", html=True), name="static")
