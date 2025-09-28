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
async def test_prediction_incorrecte():
    # 2) résultat volontairement faux à ne PAS obtenir
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert resp.status_code == 200
    assert resp.json().get("predictions") != [-9999]  # valeur absurde, doit échouer si renvoyée

@pytest.mark.anyio
async def test_json_malforme():
    # 3) JSON incorrect (exemple demandé : {[3.5, 1.2, 4.9]})
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post(
            "/predict",
            content='{[3.5, 1.2, 4.9]}',  # JSON invalide
            headers={"Content-Type": "application/json"}
        )
    assert resp.status_code in (400, 422)
