# tests/test_api.py
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.anyio
async def test_predict_success():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/predict", json={
        "features": [1.0, 2.0, 3.0]
    })
    assert resp.status_code == 200
    assert {"predictions": [2.0, 4.0, 6.0]} == resp.json()

@pytest.mark.anyio
async def test_predict_unprocessable_entity():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/predict", json={
        "feature1": 1.0,
        "feature2": 2.0,
        "feature3": 3.0
    })
    assert resp.status_code == 422
