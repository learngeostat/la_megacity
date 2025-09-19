from flask import Flask
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello from LA Megacity - Phase 1 Test!'

@app.route('/health')
def health():
    return {'status': 'healthy', 'phase': 'phase-1'}, 200

@app.route('/info')
def info():
    return {
        'port': os.environ.get('PORT', 'not-set'),
        'environment': dict(os.environ),
        'phase': 'phase-1-flask'
    }, 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Phase 1 test on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
