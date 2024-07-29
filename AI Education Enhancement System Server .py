#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Server File


# In[ ]:


from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from urllib.parse import parse_qs, urlparse

class RequestHandler(BaseHTTPRequestHandler):
    def _send_response(self, data, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def do_GET(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == '/all_recommendations':
            # Simulate response
            data = {"message": "All recommendations"}
            self._send_response(data)
        elif parsed_path.path == '/previous_recommendations':
            # Simulate response
            data = {"message": "Previous recommendations"}
            self._send_response(data)
        elif parsed_path.path == '/most_searched_recommendations':
            # Simulate response
            data = {"message": "Most searched recommendations"}
            self._send_response(data)
        elif parsed_path.path == '/random_recommendations':
            # Simulate response
            data = {"message": "Random recommendations"}
            self._send_response(data)
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = parse_qs(post_data.decode('utf-8'))

        student_id = data.get('student_id', [None])[0]
        student_name = data.get('student_name', [None])[0]

        # Simulate response
        response = {
            "recommendation": f"Recommendation for student ID {student_id} or name {student_name}"
        }
        self._send_response(response)

def run(server_class=HTTPServer, handler_class=RequestHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting httpd server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()

