from fastapi.testclient import TestClient
from backend.api import app

client = TestClient(app)

def test_cors_configuration_valid():
    # Simulate a VALID preflight request
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type, Authorization"
    }
    response = client.options("/", headers=headers)

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"

    allow_methods = response.headers.get("access-control-allow-methods", "")
    print(f"Valid Request - Allowed Methods: {allow_methods}")

    allow_headers = response.headers.get("access-control-allow-headers", "")
    print(f"Valid Request - Allowed Headers: {allow_headers}")

    allowed_methods_list = [m.strip() for m in allow_methods.split(',')]
    assert "GET" in allowed_methods_list
    assert "POST" in allowed_methods_list
    assert "PUT" not in allowed_methods_list
    assert "PATCH" not in allowed_methods_list

    # Check headers
    assert "Content-Type" in allow_headers
    assert "Authorization" in allow_headers

def test_cors_configuration_invalid_header():
    # Simulate an INVALID preflight request (bad header)
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type, X-Custom-Bad-Header"
    }
    response = client.options("/", headers=headers)

    # Should be rejected because X-Custom-Bad-Header is not allowed
    print(f"Invalid Request Status: {response.status_code}")
    assert response.status_code == 400

if __name__ == "__main__":
    test_cors_configuration_valid()
    test_cors_configuration_invalid_header()
    print("All CORS tests passed!")
