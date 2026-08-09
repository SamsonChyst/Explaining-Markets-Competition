"""Modal deployment for the Explaining Markets starter.

This is plumbing — you shouldn't need to edit it. It defines a small FastAPI app
and deploys it as a persistent, public web endpoint:

    GET  /    health check
    POST /    receive a signed event, verify, predict, submit
              (POST /competition/webhook is kept as an alias of the same handler)

The webhook is served at the root path on purpose: the URL Modal prints on deploy
*is* your webhook URL — paste it into the portal as-is, nothing to append.

Deploy:    uv run modal deploy modal_app.py
Dev/local: uv run modal serve modal_app.py
"""

import modal
app = modal.App("explaining-markets-starter")

image = (
    modal.Image.debian_slim()
    .pip_install("fastapi[standard]", "httpx", "mistralai>=2.7.1", "python-dotenv")
    .add_local_python_source("explaining_markets", "predict")
)

seen_webhooks = modal.Dict.from_name("em-webhook-dedupe", create_if_missing=True)
secrets = [modal.Secret.from_dotenv(__file__)]


def _claim(webhook_id):
    if not webhook_id:
        return True
    return seen_webhooks.put(webhook_id, "in_flight", skip_if_exists=True)


async def _claim_aio(webhook_id):
    if not webhook_id:
        return True
    return await seen_webhooks.put.aio(
        webhook_id,
        "in_flight",
        skip_if_exists=True,
    )


def _release(webhook_id, submitted):
    if not webhook_id:
        return

    if submitted:
        seen_webhooks[webhook_id] = "done"
    else:
        seen_webhooks.pop(webhook_id, None)


@app.function(
    image=image,
    secrets=secrets,
    timeout=600,
    retries=0,
)
def predict_and_submit(event: dict, webhook_id: str | None = None):
    from explaining_markets.client import submit_predictions
    from explaining_markets.config import Config
    from explaining_markets.event_utils import is_test, neutral_predictions
    from predict import predict

    submitted = False

    try:
        predictions = (
            neutral_predictions(event)
            if is_test(event)
            else predict(event)
        )

        submit_predictions(
            event_id=event["event_id"],
            predictions=predictions,
            config=Config.from_env(),
        )

        submitted = True

    except Exception as exc:
        print(
            f"[ERROR] prediction failed for event "
            f"{event.get('event_id')}: {exc}"
        )

    finally:
        _release(webhook_id, submitted)


@app.function(
    image=image,
    secrets=secrets,
)
@modal.asgi_app(label="explaining-markets")
def web():
    from fastapi import FastAPI, Request, Response

    from explaining_markets import WebhookVerificationError, verify_webhook
    from explaining_markets.config import Config
    from explaining_markets.event_utils import log_deadline

    api = FastAPI(title="Explaining Markets starter")

    @api.get("/")
    def health() -> dict:
        return {
            "ok": True,
            "service": "explaining-markets-starter",
        }

    @api.post("/")
    @api.post("/competition/webhook")
    async def competition_webhook(request: Request) -> Response:
        config = Config.from_env()

        raw_body = await request.body()

        print("=== INCOMING WEBHOOK BODY ===")
        print(raw_body.decode("utf-8", errors="ignore"))
        print("=============================")

        try:
            event = verify_webhook(
                raw_body=raw_body,
                headers=request.headers,
                secret=config.webhook_secret,
            )
        except WebhookVerificationError as exc:
            return Response(
                content=str(exc),
                status_code=401,
            )

        webhook_id = event.get("id")

        if not await _claim_aio(webhook_id):
            return Response(status_code=200)

        log_deadline(event)

        await predict_and_submit.spawn.aio(
            event,
            webhook_id,
        )

        return Response(status_code=200)

    return api