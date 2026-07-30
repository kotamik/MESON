#!/usr/bin/env python3
"""tiny static server for the sandbox. es modules need http, not file://.

    py serve.py            # serves http://localhost:8000
    py serve.py 9000       # custom port
"""
import http.server
import socketserver
import sys
import os

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
os.chdir(os.path.dirname(os.path.abspath(__file__)))


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # don't let the browser cache modules while hacking on this
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"sun-earth-moon sandbox  ->  http://localhost:{PORT}")
    print("ctrl+c to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

# so long, and thanks for all the fish