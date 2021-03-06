from test_model import (
	load_model,
	get_prediction
)

from http.server import HTTPServer, BaseHTTPRequestHandler
import ssl
import json
import traceback
import os

# server config
PORT = 3000
SSL_KEY = "ssl/private.key"
SSL_CERT = "ssl/ca_bundle.crt"


config, tokenizer, model = load_model("chess/bert-base-multilingual-uncased_English_translated_baseline_32/")

class ModelRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, *args):
        super().__init__(*args)

    def do_GET(self):
        try:
            if self.path == "/":
                self.path = "/index.html"

            if self.path == "they_actually_said_that.txt":
                with open(self.path, "r") as document:
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(document.read().encode("utf-8"))

            if self.path[-5:] == ".html":
                with open("." + self.path, "r") as document:
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(document.read().encode("utf-8"))

            elif self.path[-3:] == ".js":
                with open("." + self.path, "r") as script:
                    self.send_response(200)
                    self.send_header("Content-type", "text/javascript")
                    self.end_headers()
                    self.wfile.write(script.read().encode("utf-8"))

            elif self.path[-4:] == ".css":
                with open("." + self.path, "r") as stylesheet:
                    self.send_response(200)
                    self.send_header("Content-type", "text/css")
                    self.end_headers()
                    self.wfile.write(stylesheet.read().encode("utf-8"))

            elif self.path[-4:] == ".svg":
                print(self.path)
                with open("." + self.path, "rb") as img:
                    print("here")
                    self.send_response(200)
                    self.send_header("Content-type", "image/svg+xml")
                    self.end_headers()
                    text = img.read()
                    self.wfile.write(text)

            else:
                with open("." + self.path, "rb") as file:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(file.read())

        except:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 not found")

    def do_POST(self):
        try:
            if self.path == "/api/parse":
                length = int(self.headers.get("content-length"))
                request = json.loads(self.rfile.read(length))

                if request["text"] == "error":
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"400 bad request")

                elif request["text"] == "not hate speech":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Vary", "Accept-Encoding, Origin")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {"verdict": 0, "confidence": 100}
                        ).encode("utf-8")
                    )

                elif request["text"] == "hate speech":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Vary", "Accept-Encoding, Origin")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {"verdict": 1, "confidence": 100}
                        ).encode("utf-8")
                    )

                else:
                    text = request["text"]
                    print(text)
                    print(request)
                    verdict, confidence = get_prediction(config, tokenizer, model, text)

                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Vary", "Accept-Encoding, Origin")
                    self.end_headers()
                    self.wfile.write(json.dumps(
                    	{"verdict": verdict, "confidence": confidence}
                    ).encode("utf-8"))
                    

            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"404 not found")

        except Exception:
            traceback.print_exc()
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"500 internal server error")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "X-PINGOTHER, Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Vary", "Accept-Encoding, Origin")
        self.end_headers()


if __name__ == "__main__":
    print("Initializing server")
    httpd = HTTPServer(("0.0.0.0", PORT), ModelRequestHandler)
    print("Wrapping SSL")
    # httpd.socket = ssl.wrap_socket(
    #     httpd.socket, keyfile=SSL_KEY, certfile=SSL_CERT, server_side=True
    # )
    print("Serving on port %d" % PORT)
    httpd.serve_forever()
